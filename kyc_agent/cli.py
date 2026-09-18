"""Terminal CLI runner for AGY-KYC Checker."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw

from .agent import run_kyc_verification_pipeline
from .database import create_submission, get_submission, init_db


def create_mock_images_if_needed(sample_dir: str = "tests/samples"):
    """Generate synthetic test images for immediate demonstration."""
    os.makedirs(sample_dir, exist_ok=True)
    ktp_sample = os.path.join(sample_dir, "sample_ktp.jpg")
    selfie_sample = os.path.join(sample_dir, "sample_selfie.jpg")

    if not os.path.exists(ktp_sample):
        # Create a mock Indonesian KTP image
        img = Image.new("RGB", (600, 380), color=(180, 215, 245))
        draw = ImageDraw.Draw(img)
        # Card header
        draw.text((160, 20), "PROVINSI DKI JAKARTA", fill=(0, 30, 80))
        draw.text((180, 40), "JAKARTA SELATAN", fill=(0, 30, 80))
        # NIK & Fields
        draw.text((40, 80), "NIK          : 3171021809950003", fill=(0, 0, 0))
        draw.text((40, 110), "Nama         : BUDI SANTOSO", fill=(0, 0, 0))
        draw.text((40, 140), "Tempat/Tgl   : JAKARTA, 18-09-1995", fill=(0, 0, 0))
        draw.text((40, 170), "Jenis Kelamin: LAKI-LAKI", fill=(0, 0, 0))
        draw.text((40, 200), "Alamat       : JL. JENDERAL SUDIRMAN", fill=(0, 0, 0))
        draw.text((40, 230), "Agama        : KATOLIK", fill=(0, 0, 0))
        draw.text((40, 260), "Pekerjaan    : KARYAWAN SWASTA", fill=(0, 0, 0))
        draw.text((40, 290), "Kewarganegaraan: WNI", fill=(0, 0, 0))
        draw.text((40, 320), "Berlaku Hingga : SEUMUR HIDUP", fill=(0, 0, 0))
        # Photo box on the right
        draw.rectangle([(420, 90), (550, 260)], fill=(210, 60, 60), outline=(0, 0, 0))
        # Mock face oval
        draw.ellipse([(445, 120), (525, 230)], fill=(245, 215, 185), outline=(0, 0, 0))
        img.save(ktp_sample, "JPEG", quality=95)

    if not os.path.exists(selfie_sample):
        # Create a matching mock selfie image
        img = Image.new("RGB", (400, 500), color=(235, 240, 245))
        draw = ImageDraw.Draw(img)
        draw.rectangle([(80, 80), (320, 420)], fill=(240, 240, 240))
        draw.ellipse([(120, 140), (280, 360)], fill=(245, 215, 185), outline=(0, 0, 0))
        draw.text((140, 440), "LIVE SELFIE", fill=(50, 50, 50))
        img.save(selfie_sample, "JPEG", quality=95)

    return ktp_sample, selfie_sample


async def main_cli():
    parser = argparse.ArgumentParser(description="AGY-KYC Checker CLI Verification Runner")
    parser.add_argument("--mock", action="store_true", help="Force mock testing mode")
    parser.add_argument("--ktp", type=str, help="Path to Indonesian KTP image")
    parser.add_argument("--selfie", type=str, help="Path to applicant live selfie")
    args = parser.parse_args()

    if args.mock:
        os.environ["MOCK_MODE"] = "1"

    print("=" * 60)
    print("🇮🇩  AGY-KYC Checker - Identity Verification Pipeline")
    print("=" * 60)

    ktp_path = args.ktp
    selfie_path = args.selfie

    if not ktp_path or not selfie_path:
        print("ℹ️  No image paths specified. Generating / using synthetic test samples...")
        ktp_path, selfie_path = create_mock_images_if_needed()

    print(f"📄 KTP Image   : {ktp_path}")
    print(f"🤳 Selfie Image: {selfie_path}")

    # Initialize SQLite schema
    await init_db()

    # Create submission
    tracking_id = await create_submission(ktp_path=ktp_path, selfie_path=selfie_path)
    print(f"\n🔑 Generated Tracking ID: {tracking_id}")
    print("⏳ Executing 3-step verification pipeline...")

    # Run pipeline
    result = await run_kyc_verification_pipeline(tracking_id)

    # Fetch stored record
    record = await get_submission(tracking_id)

    print("\n" + "=" * 60)
    print(f"📊 VERIFICATION RESULT: {record.get('status')}")
    print("=" * 60)
    print(f"• Tracking ID     : {record.get('tracking_id')}")
    print(f"• Full Name       : {record.get('nama')}")
    print(f"• NIK             : {record.get('nik')}")
    print(f"• Biometric Match : {record.get('similarity_score', 0):.1%}")
    print(f"• Cropped Face    : {record.get('crop_path')}")
    print(f"• Duplicate Alert : {'YES (Sybil Warning!)' if record.get('is_duplicate_face') else 'No (Unique)'}")
    print(f"• AI Verdict Note : {record.get('ai_explanation')}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main_cli())
