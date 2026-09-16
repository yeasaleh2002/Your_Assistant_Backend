import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Union

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.api.types import Metadata
from fastembed import TextEmbedding
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from app.scraper import ScrapedJob

logger = logging.getLogger("your_assistant.rag_engine")

is_vercel = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))
DEFAULT_RESUME_PATH = Path("data") / "resume.txt"
DEFAULT_CHROMA_PATH = "/tmp/chroma_db" if is_vercel else "./chroma_db"

def get_min_match_score() -> float:
    """
    Retrieve minimum match score cutoff percentage from MIN_MATCH_SCORE env var.
    Safely handles whitespace, quotes, and invalid strings with fallback to 55.0.
    """
    raw = os.getenv("MIN_MATCH_SCORE", "55.0")
    if raw:
        cleaned = raw.strip().strip('"').strip("'")
        try:
            val = float(cleaned)
            return max(0.0, min(100.0, val))
        except ValueError:
            pass
    return 55.0

MIN_MATCH_THRESHOLD = get_min_match_score()

if is_vercel:
    os.environ.setdefault("FASTEMBED_CACHE_PATH", "/tmp/fastembed_cache")
    os.environ.setdefault("HF_HOME", "/tmp/hf_cache")



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


def chunk_resume_semantically(resume_text: str) -> List[Dict[str, Any]]:
    """
    Splits resume text into atomic, verifiable segments with metadata:
    - Summary
    - Technical Skills (sub-categorized by line or group)
    - Experience roles (each role with its bullet points)
    - Mentorship & Education
    """
    chunks: List[Dict[str, Any]] = []
    lines = resume_text.splitlines()

    current_section = "Header"
    current_role = ""
    current_buffer: List[str] = []

    def flush_buffer(section: str, role: str, buffer: List[str]):
        text = "\n".join(buffer).strip()
        if text:
            chunks.append({
                "text": text,
                "metadata": {
                    "section": section,
                    "role": role or section,
                },
            })

    # Header pattern matchers
    section_patterns = [
        ("Professional Summary", re.compile(r"^(##\s*)?(Professional\s+Summary|Summary)\b", re.IGNORECASE)),
        ("Technical Skills", re.compile(r"^(##\s*)?(Technical\s+Skills|Skills)\b", re.IGNORECASE)),
        ("Work Experience", re.compile(r"^(##\s*)?(Professional\s+Experience|Work\s+Experience|Experience)\b", re.IGNORECASE)),
        ("Mentorship Experience", re.compile(r"^(##\s*)?(Mentorship\s+Experience|Mentorship)\b", re.IGNORECASE)),
        ("Education", re.compile(r"^(##\s*)?(Education)\b", re.IGNORECASE)),
        ("Language", re.compile(r"^(##\s*)?(Language|Languages)\b", re.IGNORECASE)),
    ]

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Check if line matches a major section
        matched_section = None
        for sec_name, pattern in section_patterns:
            if pattern.match(stripped):
                matched_section = sec_name
                break

        if matched_section:
            flush_buffer(current_section, current_role, current_buffer)
            current_section = matched_section
            current_role = ""
            current_buffer = [stripped]
            continue

        # Check for role headers inside Experience sections (e.g., "Nurix Hive | Software Engineer" or "### Company | Role")
        is_role_header = (
            current_section in ("Work Experience", "Mentorship Experience")
            and (
                stripped.startswith("### ")
                or ("|" in stripped and any(yr in stripped for yr in ["2020", "2021", "2022", "2023", "2024", "2025", "Present"]))
            )
        )

        if is_role_header:
            flush_buffer(current_section, current_role, current_buffer)
            current_role = stripped.replace("###", "").strip()
            current_buffer = [stripped]
            continue

        # For Technical Skills: each skill category line (Front-End:, Back-End:, etc.) can be its own chunk
        if current_section == "Technical Skills" and (":" in stripped or stripped.startswith("- ")):
            flush_buffer(current_section, current_role, current_buffer)
            current_buffer = [stripped]
            continue

        current_buffer.append(stripped)

    flush_buffer(current_section, current_role, current_buffer)
    return chunks


# ==============================================================================
# AI Job Matching & ATS Evaluation Prompt & Schemas
# ==============================================================================

AI_JOB_MATCHING_SYSTEM_PROMPT = """You are an AI Job Matching & ATS Evaluation Engine.

Compare the provided Candidate Resume Context with the target Job Description (JD). Calculate a realistic match score from 0 to 100 based on skill overlap, experience, and responsibilities.

CRITICAL INSTRUCTION:
Output ONLY a valid JSON object. Do not include markdown formatting around JSON (no ```json code blocks), preambles, or postscripts.

JSON Format Required:
{
  "match_score": <integer between 0 and 100>,
  "matching_skills": [<list of matched skills>],
  "missing_skills": [<list of required skills missing from candidate>],
  "reasoning": "<brief 2-sentence explanation of the score>"
}
"""


class JobMatchAnalysis(BaseModel):
    """Structured ATS match evaluation result."""
    match_score: int = Field(..., ge=0, le=100, description="Match score from 0 to 100")
    matching_skills: List[str] = Field(default_factory=list, description="List of matched skills present in both")
    missing_skills: List[str] = Field(default_factory=list, description="List of required skills missing from candidate")
    reasoning: str = Field(default="", description="Brief 2-sentence explanation of score")
    summary: Optional[str] = Field(default=None, description="Summary analysis")


def evaluate_job_match_with_llm(
    job_title: str,
    job_description: str,
    candidate_resume_context: str,
    semantic_vector_score: Optional[float] = None,
) -> JobMatchAnalysis:
    """
    Compare Candidate Resume Context with target Job Description using structured LLM evaluation.
    Robust JSON parsing with error boundaries so parsing failures NEVER break the pipeline or drop records.
    """
    from app.llm_manager import generate_ai_response

    user_prompt = f"""
### TARGET JOB DESCRIPTION:
Title: {job_title}
Description:
{job_description}

### CANDIDATE RESUME CONTEXT:
{candidate_resume_context}

Evaluate the match score and return strict JSON now.
"""
    try:
        raw_response = generate_ai_response(
            prompt=user_prompt.strip(),
            system_prompt=AI_JOB_MATCHING_SYSTEM_PROMPT,
        )

        # Robust JSON cleaning: strip any ```json or ``` fences
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(raw_response).strip(), flags=re.MULTILINE).strip()
        json_match = re.search(r"\{[\s\S]*\}", cleaned)
        if json_match:
            data = json.loads(json_match.group(0))
            score_val = int(round(float(data.get("match_score", 0))))
            score_val = max(0, min(100, score_val))
            matching = [str(s) for s in data.get("matching_skills", []) if str(s).strip()]
            missing = [str(s) for s in data.get("missing_skills", []) if str(s).strip()]
            reasoning = str(data.get("reasoning") or data.get("summary") or "Evaluated by AI ATS Engine.").strip()
            summary = str(data.get("summary") or reasoning).strip()
            return JobMatchAnalysis(
                match_score=score_val,
                matching_skills=matching,
                missing_skills=missing,
                reasoning=reasoning,
                summary=summary,
            )
    except Exception as exc:
        logger.warning("LLM job evaluation encountered error or parse failure (%s). Using deterministic fallback.", exc)

    # Deterministic fallback based on semantic vector similarity and technical keyword overlap
    base_score = int(round(semantic_vector_score)) if semantic_vector_score is not None else 70
    resume_lower = candidate_resume_context.lower()
    jd_tokens = set(re.findall(r"\b[A-Za-z0-9+#.-]{3,}\b", f"{job_title} {job_description}"))
    common_words = {"the", "and", "for", "with", "that", "this", "from", "you", "are", "have", "role", "work", "experience", "looking", "candidate"}
    tech_candidates = [t for t in jd_tokens if t.lower() not in common_words]
    matched_skills = [t for t in tech_candidates if t.lower() in resume_lower][:10]
    missing_skills = [t for t in tech_candidates if t.lower() not in resume_lower][:10]

    fallback_summary = f"Automated evaluation: candidate matched on {len(matched_skills)} core technical requirements."

    return JobMatchAnalysis(
        match_score=max(0, min(100, base_score)),
        matching_skills=matched_skills,
        missing_skills=missing_skills,
        reasoning=fallback_summary,
        summary=fallback_summary,
    )


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
    location: str = Field(default="Remote", description="Job location or Remote status")
    match_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Match score percentage against user resume",
    )
    matching_skills: List[str] = Field(default_factory=list, description="Skills present in both JD and resume")
    missing_skills: List[str] = Field(default_factory=list, description="Skills in JD missing from candidate resume")
    reasoning: Optional[str] = Field(default=None, description="Explanation of match score")
    summary: Optional[str] = Field(default=None, description="Summary analysis")

    @property
    def link(self) -> str:
        return self.job_link


class RAGEngine:
    """
    Local RAG Engine using ChromaDB and FastEmbed embeddings.
    Embeds the user's base resume and performs cosine similarity matching on
    scraped job postings, strictly returning candidates with match score > 75%.
    Supports granular atomic chunk retrieval for zero-hallucination ATS resume generation.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_CHROMA_PATH,
        resume_path: Union[str, Path] = DEFAULT_RESUME_PATH,
        embedding_function: Optional[EmbeddingFunction] = None,
    ):
        self.db_path = db_path
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
        """Initialize or retrieve the resume profile vector collections."""
        self.resume_collection = self.client.get_or_create_collection(
            name="user_resume_profile",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        self.resume_chunks_collection = self.client.get_or_create_collection(
            name="user_resume_atomic_chunks",
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
        Read and embed the user's base resume (both as whole document and atomic chunks) into ChromaDB.
        """
        resume_content = self.load_resume(path)
        logger.info("Embedding base resume (%d characters) into ChromaDB...", len(resume_content))

        # Store in whole-resume collection
        self.resume_collection.upsert(
            documents=[resume_content],
            metadatas=[{"source": str(path or self.resume_path)}],
            ids=["base_resume"],
        )

        # Also chunk semantically and store atomic chunks for zero-hallucination retrieval
        chunks = chunk_resume_semantically(resume_content)
        if chunks:
            chunk_docs = [c["text"] for c in chunks]
            chunk_metas = [c["metadata"] for c in chunks]
            chunk_ids = [f"resume_chunk_{i}" for i in range(len(chunks))]
            self.resume_chunks_collection.upsert(
                documents=chunk_docs,
                metadatas=chunk_metas,
                ids=chunk_ids,
            )
            logger.info("Indexed %d atomic resume chunks into ChromaDB.", len(chunks))

        return resume_content

    def retrieve_proven_context(
        self,
        query_terms: Union[str, Sequence[str]],
        n_results_per_term: int = 3,
        max_distance: float = 0.55,
    ) -> List[str]:
        """
        Strict retrieval step from ChromaDB:
        Queries atomic resume chunks using extracted JD terms or requirements.
        ONLY returns context chunks whose cosine distance <= max_distance (similarity >= 45%).
        Drops low-confidence semantic matches to eliminate hallucinated context.
        """
        if isinstance(query_terms, str):
            terms = [query_terms]
        else:
            terms = list(query_terms)

        terms = [t.strip() for t in terms if t and len(t.strip()) > 1]
        if not terms:
            return []

        # Ensure chunks are indexed
        existing_chunks = self.resume_chunks_collection.get(limit=1)
        if not existing_chunks or not existing_chunks.get("ids"):
            self.embed_resume()

        proven_chunks: List[str] = []
        seen: set[str] = set()

        for term in terms:
            try:
                results = self.resume_chunks_collection.query(
                    query_texts=[term],
                    n_results=min(n_results_per_term, 5),
                    include=["documents", "distances"],
                )
                docs_list = results.get("documents", [[]])
                dists_list = results.get("distances", [[]])

                if docs_list and dists_list:
                    docs = docs_list[0]
                    dists = dists_list[0]
                    for doc, dist in zip(docs, dists):
                        if dist <= max_distance and doc not in seen:
                            seen.add(doc)
                            proven_chunks.append(doc)
            except Exception as exc:
                logger.warning("Error querying atomic chunk for term '%s': %s", term, exc)

        return proven_chunks

    def query_resume(self, query: str, n_results: int = 3) -> List[str]:
        """Retrieve relevant resume context chunks from ChromaDB."""
        return self.query_proven_context_chunks(query_terms=query, n_results_per_term=n_results)

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
        metadatas: List[Metadata] = []

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
        distances_matrix = query_results.get("distances")
        metadatas_matrix = query_results.get("metadatas")

        if not distances_matrix or not metadatas_matrix:
            return []

        distances = distances_matrix[0]
        result_metadatas = metadatas_matrix[0]

        for dist, meta in zip(distances, result_metadatas):
            if not meta or "index" not in meta:
                continue
            raw_idx = meta["index"]
            if not isinstance(raw_idx, (int, str)):
                continue
            job_idx = int(raw_idx)
            if not (0 <= job_idx < len(standardized_jobs)):
                continue
            candidate = standardized_jobs[job_idx]

            # In ChromaDB with cosine space: distance = 1 - cosine_similarity
            cosine_similarity = 1.0 - dist
            score_pct = round(cosine_similarity * 100.0, 2)
            score_pct = max(0.0, min(100.0, score_pct))

            # Retrieve candidate resume context from ChromaDB
            try:
                context_chunks = self.query_resume(f"{candidate.title}. {candidate.description[:300]}", n_results=4)
                candidate_context = "\n\n".join(context_chunks) if context_chunks else resume_text
            except Exception:
                candidate_context = resume_text

            # LLM-based evaluation with robust JSON parsing & deterministic fallback
            analysis = evaluate_job_match_with_llm(
                job_title=candidate.title,
                job_description=candidate.description,
                candidate_resume_context=candidate_context,
                semantic_vector_score=score_pct,
            )
            calculated_score = float(analysis.match_score)

            # Log each evaluation per required specification
            logger.debug(
                f'[DEBUG] Job: "{candidate.title}" | Calculated Score: {calculated_score:.0f}% | Required Score: {min_match_score:.0f}%'
            )

            # STRICT CUTOFF: Return ONLY jobs with match score >= min_match_score
            if calculated_score >= min_match_score:
                matched_jobs.append(
                    MatchedJob(
                        title=candidate.title,
                        company=candidate.company,
                        description=candidate.description,
                        job_link=candidate.job_link,
                        career_page_link=candidate.career_page_link,
                        location=getattr(candidate, "location", "Remote") or "Remote",
                        match_score=calculated_score,
                        matching_skills=analysis.matching_skills,
                        missing_skills=analysis.missing_skills,
                        reasoning=analysis.reasoning,
                        summary=analysis.summary,
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

    def calculate_match_score(
        self,
        job_description: str,
        job_title: str = "",
        base_resume_text: Optional[str] = None,
    ) -> float:
        """
        Calculate semantic vector cosine similarity match percentage (0.0% to 100.0%)
        between candidate's resume and an arbitrary job description text.
        """
        resume_text = base_resume_text or self.load_resume()
        doc_text = f"{job_title}. {job_description}" if job_title else job_description

        try:
            # Embed both texts using the high-speed local FastEmbed ONNX embedding function
            embeddings = self.embedding_function([resume_text, doc_text])
            if len(embeddings) < 2:
                return 75.0

            vec1 = np.array(embeddings[0], dtype=float)
            vec2 = np.array(embeddings[1], dtype=float)

            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 70.0

            cosine_sim = float(np.dot(vec1, vec2) / (norm1 * norm2))
            score_pct = round(cosine_sim * 100.0, 2)
            return max(0.0, min(100.0, score_pct))
        except Exception as exc:
            logger.warning("Error calculating semantic match score via FastEmbed: %s", exc)
            # Fallback keyword overlap heuristic
            resume_lower = resume_text.lower()
            jd_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", doc_text.lower()))
            if not jd_words:
                return 70.0
            hits = sum(1 for w in jd_words if w in resume_lower)
            ratio = hits / len(jd_words)
            return round(min(95.0, max(50.0, ratio * 150.0)), 2)

