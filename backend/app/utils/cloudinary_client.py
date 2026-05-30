import cloudinary
import cloudinary.uploader
import httpx
import tempfile
import os
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def configure_cloudinary():
    cloudinary.config(
        cloud_name=settings.CLOUDINARY_CLOUD_NAME,
        api_key=settings.CLOUDINARY_API_KEY,
        api_secret=settings.CLOUDINARY_API_SECRET,
        secure=True,
    )


def upload_pdf(file_bytes: bytes, filename: str, user_id: int) -> dict:
    """Upload PDF to Cloudinary and return url + public_id."""
    configure_cloudinary()
    result = cloudinary.uploader.upload(
        file_bytes,
        resource_type="raw",
        folder=f"papermind/{user_id}",
        public_id=Path(filename).stem,
        format="pdf",
        overwrite=False,
        use_filename=True,
    )
    logger.info("cloudinary_upload", public_id=result["public_id"], url=result["secure_url"])
    return {"url": result["secure_url"], "public_id": result["public_id"]}


async def download_pdf_to_temp(url: str) -> str:
    """Download a PDF from Cloudinary URL to a temp file, return the temp path."""
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(url)
        response.raise_for_status()

    suffix = ".pdf"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(response.content)
    tmp.close()
    logger.info("cloudinary_downloaded", tmp=tmp.name, size=len(response.content))
    return tmp.name


def delete_pdf(public_id: str):
    configure_cloudinary()
    try:
        cloudinary.uploader.destroy(public_id, resource_type="raw")
        logger.info("cloudinary_deleted", public_id=public_id)
    except Exception as e:
        logger.warning("cloudinary_delete_failed", public_id=public_id, error=str(e))
