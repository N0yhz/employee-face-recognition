"""CLI tool to register an employee's face into the recognition database.

Usage::

    python register_employee.py --image path/to/face.jpg --id EMP001 --name "John Doe"
"""

import argparse
import sys

import cv2

import config
from database.employee_db import EmployeeDatabase
from recognition.embedder import FaceEmbedder


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Register an employee face into the recognition database."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the employee face image (JPEG/PNG).",
    )
    parser.add_argument(
        "--id",
        dest="employee_id",
        required=True,
        help="Unique employee ID (e.g. EMP001).",
    )
    parser.add_argument(
        "--name",
        required=True,
        help='Employee display name (e.g. "John Doe").',
    )
    parser.add_argument(
        "--db",
        default=config.DB_PATH,
        help=f"Path to the employee database file (default: {config.DB_PATH}).",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the employee registration CLI."""
    args = parse_args()

    # Load image
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"[register_employee] ERROR: Could not read image '{args.image}'.")
        sys.exit(1)

    print(f"[register_employee] Loaded image: {args.image}")

    # Initialise embedder
    print("[register_employee] Initialising ArcFace embedder…")
    try:
        embedder = FaceEmbedder(model_name=config.ARCFACE_MODEL)
    except Exception as exc:
        print(f"[register_employee] ERROR: Failed to initialise embedder: {exc}")
        sys.exit(1)

    # Extract embedding from the full image (no bbox crop — use entire image)
    h, w = frame.shape[:2]
    embedding = embedder.get_embedding(frame, [0, 0, w, h])
    if embedding is None:
        print(
            "[register_employee] ERROR: No face detected in the provided image. "
            "Make sure the image clearly shows a single face."
        )
        sys.exit(1)

    print(f"[register_employee] Embedding extracted (dim={embedding.shape[0]}).")

    # Save to database
    db = EmployeeDatabase(db_path=args.db)
    db.register(employee_id=args.employee_id, name=args.name, embedding=embedding)
    print(
        f"[register_employee] Registered employee '{args.name}' "
        f"(ID: {args.employee_id}) in '{args.db}'."
    )

    # Show current employees
    employees = db.list_employees()
    print(f"[register_employee] Total employees in DB: {len(employees)}")
    for emp in employees:
        print(f"  • {emp['id']}: {emp['name']}")


if __name__ == "__main__":
    main()
