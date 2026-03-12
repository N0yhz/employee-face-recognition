"""Employee face embedding storage and lookup using JSON persistence."""

import json
import os
from typing import Optional

import numpy as np


class EmployeeDatabase:
    """Manages employee face embeddings with JSON-backed persistence.

    Embeddings are stored as plain lists in JSON so the file is human-readable
    and portable.  They are deserialized back to :class:`numpy.ndarray` on load.
    """

    def __init__(self, db_path: str) -> None:
        """Initialise the database, loading an existing file if one exists.

        Args:
            db_path: Path to the JSON file used for persistence.
        """
        self.db_path = db_path
        # Internal store: {employee_id: {"name": str, "embedding": np.ndarray}}
        self._employees: dict = {}
        if os.path.exists(db_path):
            self.load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, employee_id: str, name: str, embedding: np.ndarray) -> None:
        """Register or update an employee's face embedding.

        Args:
            employee_id: Unique identifier for the employee (e.g. ``"EMP001"``).
            name: Human-readable display name.
            embedding: 512-dimensional ArcFace embedding as a NumPy array.
        """
        self._employees[employee_id] = {
            "name": name,
            "embedding": embedding,
        }
        self.save()

    def find_match(
        self, embedding: np.ndarray, threshold: float
    ) -> Optional[dict]:
        """Find the closest registered employee using cosine similarity.

        Args:
            embedding: Query face embedding (512-dim, L2-normalised).
            threshold: Minimum cosine similarity required to count as a match.

        Returns:
            A dict ``{"id": str, "name": str, "similarity": float}`` for the
            best match, or ``None`` when no employee exceeds *threshold*.
        """
        best_id: Optional[str] = None
        best_name: Optional[str] = None
        best_sim: float = -1.0

        for emp_id, data in self._employees.items():
            sim = float(self._cosine_similarity(embedding, data["embedding"]))
            if sim > best_sim:
                best_sim = sim
                best_id = emp_id
                best_name = data["name"]

        if best_id is not None and best_sim >= threshold:
            return {"id": best_id, "name": best_name, "similarity": best_sim}
        return None

    def save(self) -> None:
        """Persist the database to the JSON file.

        Embeddings are serialised as plain Python lists so they can be stored
        in standard JSON.  The parent directory is created automatically when
        it does not yet exist.
        """
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
        serialisable: dict = {}
        for emp_id, data in self._employees.items():
            embedding = data["embedding"]
            serialisable[emp_id] = {
                "name": data["name"],
                "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
            }
        with open(self.db_path, "w", encoding="utf-8") as fh:
            json.dump(serialisable, fh)

    def load(self) -> None:
        """Load the database from the JSON file.

        Lists stored in JSON are converted back to :class:`numpy.ndarray`.
        """
        try:
            with open(self.db_path, "r", encoding="utf-8") as fh:
                raw: dict = json.load(fh)
            self._employees = {}
            for emp_id, data in raw.items():
                self._employees[emp_id] = {
                    "name": data["name"],
                    "embedding": np.array(data["embedding"], dtype=np.float32),
                }
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            print(f"[EmployeeDatabase] Failed to load '{self.db_path}': {exc}")
            self._employees = {}

    def list_employees(self) -> list:
        """Return a list of all registered employees.

        Returns:
            A list of dicts, each containing ``"id"`` and ``"name"`` keys.
        """
        return [
            {"id": emp_id, "name": data["name"]}
            for emp_id, data in self._employees.items()
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two 1-D vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Cosine similarity in the range ``[-1, 1]``.
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
