import os
from io import BytesIO
from pathlib import Path

import boto3
from dotenv import load_dotenv
from PIL import Image


class S3ImageUploader:
    """Utility class for uploading images to S3 after converting to PNG format."""

    @staticmethod
    def upload_image(image_path: str, sub_bucket: str = "") -> str:
        """
        Convert an image to PNG and upload it to S3.

        Args:
            image_path: Path to the input image file
            sub_bucket: The sub-bucket within S3 to upload the image to

        Returns:
            The URL of the uploaded image in S3

        Raises:
            FileNotFoundError: If the image file doesn't exist
            ValueError: If S3_BUCKET environment variable is not set
        """
        # Get S3 bucket from environment variable
        bucket_name = os.getenv("S3_BUCKET")
        if not bucket_name:
            raise ValueError("S3_BUCKET environment variable not set")

        # Verify image file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Open and convert image to PNG
        image = Image.open(image_path)
        if image.mode in ("RGBA", "LA", "P"):
            # Convert RGBA/LA/Palette to RGB for better compatibility
            rgb_image = Image.new("RGB", image.size, (255, 255, 255))
            rgb_image.paste(
                image, mask=image.split()[-1] if image.mode in ("RGBA", "LA") else None
            )
            image = rgb_image

        # Save PNG to bytes buffer
        png_buffer = BytesIO()
        image.save(png_buffer, format="PNG")
        png_buffer.seek(0)

        # Generate S3 key from original filename
        original_name = Path(image_path).stem
        s3_key = f"{sub_bucket}/{original_name}.png"

        # Upload to S3
        s3_client = boto3.client("s3")
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=png_buffer.getvalue(),
            ContentType="image/png",
        )

        # Return the S3 URL
        region = os.getenv("AWS_REGION") or "us-east-1"
        s3_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{s3_key}"

        return s3_url


if __name__ == "__main__":
    load_dotenv()
    img = "./data/images/0a0f2973339372b705031dbacbe3ac3341867054_Image_001.jpg"
    print(S3ImageUploader.upload_image(img))
