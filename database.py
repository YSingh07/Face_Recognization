
import os
import psycopg
from psycopg.types.json import Jsonb
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is missing in .env")

    return psycopg.connect(
        DATABASE_URL,
        connect_timeout=10
    )


def register_person(username, embeddings):
    query = """
        INSERT INTO public.face_users (username, embeddings)
        VALUES (%s, %s)
        RETURNING id;
    """

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (username, Jsonb(embeddings))
                )
                person_id = cur.fetchone()[0]

        return person_id

    except psycopg.errors.UniqueViolation:
        return None


def get_all_people():
    query = """
        SELECT username, embeddings
        FROM public.face_users
        ORDER BY username;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchall()
