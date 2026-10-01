import io
import warnings
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import HTTPException

MIME_FORMAT = {'image/jpeg': 'JPEG', 'image/png': 'PNG', 'image/webp': 'WEBP'}


def decode_image(data: bytes, content_type: str, max_pixels: int) -> Image.Image:
    if content_type not in MIME_FORMAT:
        raise HTTPException(415, 'Use a JPEG, PNG, or WebP image.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format != MIME_FORMAT[content_type]:
                    raise HTTPException(415, 'File contents do not match the declared image type.')
                if image.width * image.height > max_pixels:
                    raise HTTPException(413, 'Image dimensions exceed the pixel limit.')
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                return ImageOps.exif_transpose(image).convert('RGB')
    except HTTPException:
        raise
    except (Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise HTTPException(413, 'Image dimensions exceed the pixel limit.')
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise HTTPException(400, 'The uploaded image cannot be decoded.')
