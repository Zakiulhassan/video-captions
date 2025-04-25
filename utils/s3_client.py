import os
from minio import Minio
from minio.error import S3Error
import logging
from typing import Optional, List, Dict
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)


class MinioClient:
    def __init__(self):
        """Initialize MinIO client with environment variables."""
        self.client = Minio(
            endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
        )
        self.bucket_name = os.getenv("MINIO_BUCKET", "video-captions")
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        """Ensure the bucket exists, create if it doesn't."""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Created bucket: {self.bucket_name}")
        except S3Error as e:
            logger.error(f"Error ensuring bucket exists: {str(e)}")
            raise

    def upload_file(self, file_path: str, object_name: str) -> str:
        """
        Upload a file to MinIO.

        Args:
            file_path: Local path to the file
            object_name: Name to give the file in MinIO (including folder path)

        Returns:
            URL of the uploaded file
        """
        try:
            self.client.fput_object(self.bucket_name, object_name, file_path)
            logger.info(f"Uploaded {file_path} to {object_name}")
            return f"/{self.bucket_name}/{object_name}"
        except S3Error as e:
            logger.error(f"Error uploading file: {str(e)}")
            raise

    def download_file(self, object_name: str) -> str:
        """
        Download a file from MinIO to a temporary location.

        Args:
            object_name: Name of the file in MinIO (including folder path)

        Returns:
            Local path to the downloaded file
        """
        try:
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.close()

            # Download the file
            self.client.fget_object(self.bucket_name, object_name, temp_file.name)
            logger.info(f"Downloaded {object_name} to {temp_file.name}")
            return temp_file.name
        except S3Error as e:
            logger.error(f"Error downloading file: {str(e)}")
            raise

    def list_files(self, prefix: str = "") -> List[Dict]:
        """
        List files in a folder.

        Args:
            prefix: Folder path to list files from

        Returns:
            List of file information dictionaries
        """
        try:
            objects = self.client.list_objects(
                self.bucket_name, prefix=prefix, recursive=True
            )

            files = []
            for obj in objects:
                files.append(
                    {
                        "name": obj.object_name,
                        "size": obj.size,
                        "last_modified": obj.last_modified,
                        "url": f"/{self.bucket_name}/{obj.object_name}",
                    }
                )

            return files
        except S3Error as e:
            logger.error(f"Error listing files: {str(e)}")
            raise

    def delete_file(self, object_name: str) -> bool:
        """
        Delete a file from MinIO.

        Args:
            object_name: Name of the file in MinIO (including folder path)

        Returns:
            True if successful
        """
        try:
            self.client.remove_object(self.bucket_name, object_name)
            logger.info(f"Deleted {object_name}")
            return True
        except S3Error as e:
            logger.error(f"Error deleting file: {str(e)}")
            raise

    def get_file_url(self, object_name: str) -> str:
        """
        Get the URL for a file.

        Args:
            object_name: Name of the file in MinIO (including folder path)

        Returns:
            URL of the file
        """
        return f"/{self.bucket_name}/{object_name}"
