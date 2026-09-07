import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from fastembed import TextEmbedding
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from app.scraper import ScrapedJob

logger = logging.getLogger("your_assistant.rag_engine")

DEFAULT_RESUME_PATH = Path("data") / "resume.txt"
DEFAULT_CHROMA_PATH = "./chroma_db"
MIN_MATCH_THRESHOLD = 75.0  # Strict cutoff (> 75%) to minimize LLM token usage


# ==============================================================================
# Lightweight Open-Source Embedding Model (FastEmbed / ONNX)
# ==============================================================================

class FastEmbedEmbeddingFunction(EmbeddingFunction):
    """
    Lightweight, high-speed open-source embedding function powered by FastEmbed and ONNX Runtime.
    Uses 'BAAI/bge-small-en-v1.5' (384-dimensional dense semantic vectors).
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    @staticmethod
    def name() -> str:
        return "fastembed_bge_small"

    def get_config(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}

    @classmethod
    def build_from_config(cls, config: Dict[str, Any]) -> "FastEmbedEmbeddingFunction":
        return cls(model_name=config.get("model_name", "BAAI/bge-small-en-v1.5"))

    def __call__(self, input: Documents) -> Embeddings:
        """Embed a list of text documents into 384-dimensional normalized vectors."""
        if not input:
            return []
        safe_input = [text if text and text.strip() else "empty document" for text in input]
        embeddings = list(self._model.embed(safe_input))
        return [emb.tolist() for emb in embeddings]


# ==============================================================================
# Matched Job Schema
# ==============================================================================

class MatchedJob(BaseModel):
    """Scraped job that met or exceeded the strict semantic match score threshold (> 75%)."""
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Employer company name")
    description: str = Field(default="", description="Job description snippet")
    job_link: str = Field(..., description="Direct application or job link")
    career_page_link: str = Field(..., description="Predicted or resolved career portal URL")
    match_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Cosine similarity match score percentage against user resume",
    )


# ==============================================================================
# RAG Engine Core
# ==============================================================================

class RAGEngine:
    """
    Local Vector RAG Engine using ChromaDB.
    
    Embeds the user's base resume and performs cosine similarity matching on
    scraped job postings, strictly returning candidates with match score > 75%.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_CHROMA_PATH,
        resume_path: Union[str, Path] = DEFAULT_RESUME_PATH,
        embedding_function: Optional[EmbeddingFunction] = None,
    ):
        self.db_path = str(db_path)
        self.resume_path = Path(resume_path)
        self.embedding_function = embedding_function or FastEmbedEmbeddingFunction()

        # Initialize local persistent ChromaDB client
        os.makedirs(self.db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.db_path)
        logger.info("ChromaDB PersistentClient initialized at: %s", self.db_path)

        # Base resume cache
        self._resume_text: Optional[str] = None
        self._init_resume_collection()

    def _init_resume_collection(self) -> None:
        """Initialize or retrieve the resume profile vector collection."""
        self.resume_collection = self.client.get_or_create_collection(
            name="user_resume_profile",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def load_resume(self, path: Optional[Union[str, Path]] = None) -> str:
        """
        Read the user's base resume text from a static file.
        """
        target_path = Path(path) if path else self.resume_path
        if not target_path.exists():
            raise FileNotFoundError(f"Resume file not found at: {target_path}")

        text = target_path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Resume file at {target_path} is empty.")

        self._resume_text = text
        return text

    def embed_resume(self, path: Optional[Union[str, Path]] = None) -> str:
        """
        Read and embed the user's base resume into ChromaDB.
        """
        resume_content = self.load_resume(path)
        logger.info("Embedding base resume (%d characters) into ChromaDB...", len(resume_content))

        # Store in ChromaDB collection
        self.resume_collection.upsert(
            documents=[resume_content],
            metadatas=[{"source": str(path or self.resume_path)}],
            ids=["base_resume"],
        )
        return resume_content

    def get_resume_text(self) -> str:
        """Retrieve cached resume text or load from static file."""
        if self._resume_text is None:
            if self.resume_path.exists():
                self._resume_text = self.load_resume()
            else:
                raise FileNotFoundError(f"Base resume not found at {self.resume_path}")
        return self._resume_text

    def match_jobs(
        self,
        jobs: Sequence[Union[ScrapedJob, Dict[str, Any]]],
        min_match_score: float = MIN_MATCH_THRESHOLD,
    ) -> List[MatchedJob]:
        """
        Embed scraped job descriptions, compute cosine similarity against resume,
        and filter for jobs with match score > min_match_score (default: 75%).
        
        :param jobs: List of ScrapedJob models or dictionaries.
        :param min_match_score: Cutoff percentage threshold (e.g. 75.0).
        :return: List of MatchedJob objects exceeding the match threshold, ranked by score.
        """
        if not jobs:
            return []

        resume_text = self.get_resume_text()

        # Prepare temporary session collection for candidates to perform Cosine Similarity search
        collection_name = "job_candidates_evaluation"
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            pass

        jobs_collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

        # Standardize job objects
        standardized_jobs: List[ScrapedJob] = []
        documents: List[str] = []
        ids: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for idx, job in enumerate(jobs):
            if isinstance(job, dict):
                job_obj = ScrapedJob(**job)
            else:
                job_obj = job

            standardized_jobs.append(job_obj)

            # Document for rich semantic vector matching: Title + Description
            doc_text = f"{job_obj.title}. {job_obj.description}"
            documents.append(doc_text)
            ids.append(f"job_{idx}")
            metadatas.append({"index": idx, "title": job_obj.title, "company": job_obj.company})

        # Embed and insert candidate jobs
        jobs_collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas,
        )

        # Cosine Similarity Search using the user's resume as query
        query_results = jobs_collection.query(
            query_texts=[resume_text],
            n_results=len(jobs),
            include=["distances", "metadatas"],
        )

        matched_jobs: List[MatchedJob] = []
        distances = query_results.get("distances", [[]])[0]
        result_metadatas = query_results.get("metadatas", [[]])[0]

        for dist, meta in zip(distances, result_metadatas):
            job_idx = meta["index"]
            candidate = standardized_jobs[job_idx]

            # In ChromaDB with cosine space: distance = 1 - cosine_similarity
            cosine_similarity = 1.0 - float(dist)
            score_pct = round(cosine_similarity * 100.0, 2)
            score_pct = max(0.0, min(100.0, score_pct))

            logger.info(
                "Job Candidate: '%s' (%s) -> Cosine Distance: %.4f, Match Score: %.2f%%",
                candidate.title,
                candidate.company,
                dist,
                score_pct,
            )

            # STRICT CUTOFF: Return ONLY jobs with match score > min_match_score (75%)
            if score_pct > min_match_score:
                matched_jobs.append(
                    MatchedJob(
                        title=candidate.title,
                        company=candidate.company,
                        description=candidate.description,
                        job_link=candidate.job_link,
                        career_page_link=candidate.career_page_link,
                        match_score=score_pct,
                    )
                )

        # Sort descending by match score
        matched_jobs.sort(key=lambda j: j.match_score, reverse=True)

        logger.info(
            "RAG evaluation finished: Evaluated %d jobs, %d exceeded >%.1f%% threshold (saving LLM API costs).",
            len(jobs),
            len(matched_jobs),
            min_match_score,
        )

        # Clean up temporary evaluation collection
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            pass

        return matched_jobs
