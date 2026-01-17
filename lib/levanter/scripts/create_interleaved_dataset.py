#!/usr/bin/env python3
"""
Create and upload an interleaved image-text dataset to HuggingFace.

This script creates a new 'interleaved' split for the demo-vlm-test-dataset with:
- Single image QA samples
- Multi-image interleaved samples (with variable number of images: 2-4)
- Caption samples (simple and detailed)

All samples are shuffled together to create diverse batches.

Usage:
    python create_interleaved_dataset.py [--dry-run]
"""

import argparse
import random
from datasets import Dataset, DatasetDict, load_dataset, Features, Sequence, Image, Value


HF_DATASET = "ruili0/demo-vlm-test-dataset"
RANDOM_SEED = 42


def create_single_image_samples(image):
    """Create 6 single image QA samples (3 simple, 3 detailed)."""
    samples = []

    # Simple QA samples
    simple_qa = [
        ("What is in this image?", "A university entrance with an arched stone gateway."),
        ("What type of building is shown?", "This is the entrance to Stanford University."),
        ("Describe what you see briefly.", "A grand stone archway marking a university entrance."),
    ]

    for question, answer in simple_qa:
        samples.append({
            "messages": [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]}
            ],
            "images": [image],
            "sample_type": "single_image_simple"
        })

    # Detailed description samples
    detailed_qa = [
        (
            "Describe this image in detail.",
            "The image shows the iconic entrance to Stanford University, featuring a grand stone archway with intricate architectural details. The archway is made of sandstone and has a classic design with columns and decorative elements. Palm trees and landscaping surround the entrance, creating a welcoming atmosphere. The sky is clear and blue, suggesting a beautiful California day."
        ),
        (
            "What can you tell me about this location?",
            "This is the main entrance to Stanford University, one of the most prestigious universities in the world. The stone archway is a recognizable landmark that welcomes students, faculty, and visitors to the campus. The architecture reflects the university's Spanish colonial revival style, which is characteristic of many buildings on campus. The surrounding landscape includes mature trees and well-maintained gardens."
        ),
        (
            "Provide a comprehensive description of this scene.",
            "The photograph captures the majestic entrance gate of Stanford University in California. The centerpiece is an ornate stone arch crafted from sandstone, featuring classical architectural elements including carved details and a symmetrical design. The entrance is flanked by stone pillars and iron gates. In the background, palm trees and other vegetation create a lush, Mediterranean atmosphere. The lighting suggests it was taken during daytime with clear skies, highlighting the warm tones of the stonework."
        ),
    ]

    for question, answer in detailed_qa:
        samples.append({
            "messages": [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]}
            ],
            "images": [image],
            "sample_type": "single_image_detailed"
        })

    return samples


def create_multi_image_interleaved_samples(image):
    """Create 6 multi-image interleaved samples with variable image counts (2-4 images)."""
    samples = []

    # Pattern A: Text-Image-Text-Image-... pattern (3 samples with 2, 3, 4 images)

    # 2 images
    samples.append({
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "Here is the first image:"},
                {"type": "image"},
                {"type": "text", "text": "And here is the second image:"},
                {"type": "image"},
                {"type": "text", "text": "What do these images have in common?"}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "Both images show the Stanford University entrance gate. They appear to be the same location photographed from similar angles, featuring the iconic stone archway with its classical architectural design."}]}
        ],
        "images": [image, image],
        "sample_type": "multi_image_interleaved_2"
    })

    # 3 images
    samples.append({
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "I have three images to show you. First:"},
                {"type": "image"},
                {"type": "text", "text": "Second:"},
                {"type": "image"},
                {"type": "text", "text": "And third:"},
                {"type": "image"},
                {"type": "text", "text": "Can you describe what's consistent across all three?"}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "All three images consistently show the Stanford University entrance. The stone archway, classical architecture, and surrounding vegetation appear in each image. The lighting and perspective are similar, suggesting they may have been taken around the same time or are the same photograph."}]}
        ],
        "images": [image, image, image],
        "sample_type": "multi_image_interleaved_3"
    })

    # 4 images
    samples.append({
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "Let me show you a series of images. Image 1:"},
                {"type": "image"},
                {"type": "text", "text": "Image 2:"},
                {"type": "image"},
                {"type": "text", "text": "Image 3:"},
                {"type": "image"},
                {"type": "text", "text": "Image 4:"},
                {"type": "image"},
                {"type": "text", "text": "What's the common theme in all these images?"}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "The common theme across all four images is the Stanford University entrance gate. Each image features the distinctive stone archway that serves as the main entrance to the university. The consistent architectural elements include the sandstone construction, classical design, and the surrounding landscape with palm trees and gardens."}]}
        ],
        "images": [image, image, image, image],
        "sample_type": "multi_image_interleaved_4"
    })

    # Pattern B: Compare images pattern (3 samples with 2, 3, 4 images)

    # 2 images comparison
    samples.append({
        "messages": [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": "Compare these two images and describe any differences you notice."}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "Both images appear to show the same location - the Stanford University entrance gate. The stone archway, architectural details, and surrounding vegetation are identical in both images. Since they appear to be the same photograph, there are no noticeable differences between them."}]}
        ],
        "images": [image, image],
        "sample_type": "multi_image_compare_2"
    })

    # 3 images comparison
    samples.append({
        "messages": [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": "Look at these three images. Are they showing the same place? Explain your reasoning."}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "Yes, all three images show the same place - the entrance to Stanford University. My reasoning is based on the identical stone archway visible in each image, the same architectural style with classical elements, and the consistent surrounding landscape including palm trees. The images appear to be identical or taken from the same position."}]}
        ],
        "images": [image, image, image],
        "sample_type": "multi_image_compare_3"
    })

    # 4 images comparison
    samples.append({
        "messages": [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": "Analyze these four images. What location do they all depict and what elements are consistent?"}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "All four images depict the Stanford University entrance gate in California. The consistent elements include: 1) The ornate stone archway made of sandstone with classical architectural details, 2) The iron gates within the archway, 3) The surrounding landscape with palm trees and Mediterranean vegetation, 4) The warm tones of the stonework, and 5) The clear sky in the background. These images are identical, showing the same photograph of this iconic university landmark."}]}
        ],
        "images": [image, image, image, image],
        "sample_type": "multi_image_compare_4"
    })

    return samples


def create_caption_samples(image):
    """Create 6 caption samples (3 simple, 3 detailed)."""
    samples = []

    # Simple captions
    simple_captions = [
        ("Caption this image briefly.", "Stanford University entrance gate on a sunny day."),
        ("Write a short caption for this photo.", "The iconic stone archway at Stanford University."),
        ("Give me a one-line description.", "A grand stone entrance to a prestigious California university."),
    ]

    for prompt, caption in simple_captions:
        samples.append({
            "messages": [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": caption}]}
            ],
            "images": [image],
            "sample_type": "caption_simple"
        })

    # Detailed captions
    detailed_captions = [
        (
            "Write a detailed caption for this image.",
            "The majestic entrance to Stanford University stands proudly under a clear blue sky. The ornate stone archway, crafted from warm-toned sandstone, features classical architectural elements that reflect the university's Spanish colonial revival style. Lush palm trees and carefully maintained gardens frame the iconic gateway, creating an inviting entrance to one of the world's most prestigious academic institutions."
        ),
        (
            "Provide a comprehensive caption that captures the essence of this scene.",
            "Bathed in California sunshine, the Stanford University main entrance presents a timeless image of academic prestige. The hand-carved stone arch rises elegantly, its classical details telling stories of over a century of educational excellence. Mediterranean landscaping with towering palms softens the grandeur of the stonework, while the clear sky above promises endless possibilities for the students who pass through this historic gateway."
        ),
        (
            "Create an evocative caption for this photograph.",
            "Where dreams meet opportunity: The Stanford University entrance gate welcomes visitors with its stunning sandstone architecture and stately presence. This iconic landmark, framed by swaying palms and manicured grounds, has witnessed generations of scholars, innovators, and leaders begin their transformative journeys. The arch stands as both a physical and symbolic threshold to world-class education."
        ),
    ]

    for prompt, caption in detailed_captions:
        samples.append({
            "messages": [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": caption}]}
            ],
            "images": [image],
            "sample_type": "caption_detailed"
        })

    return samples


def create_interleaved_dataset():
    """Create the full interleaved dataset with shuffled samples."""
    print("Loading existing dataset to get images...")
    existing_ds = load_dataset(HF_DATASET)

    # Get the image from the single_image split
    image = existing_ds["single_image"][0]["images"][0]
    print(f"Loaded image: {type(image)}")

    # Create all samples
    print("Creating samples...")
    all_samples = []
    all_samples.extend(create_single_image_samples(image))
    all_samples.extend(create_multi_image_interleaved_samples(image))
    all_samples.extend(create_caption_samples(image))

    print(f"Created {len(all_samples)} samples total")

    # Count by type
    type_counts = {}
    for sample in all_samples:
        sample_type = sample["sample_type"]
        type_counts[sample_type] = type_counts.get(sample_type, 0) + 1
    print("Sample types before shuffle:")
    for sample_type, count in sorted(type_counts.items()):
        print(f"  {sample_type}: {count}")

    # Shuffle the samples
    random.seed(RANDOM_SEED)
    random.shuffle(all_samples)
    print(f"\nShuffled {len(all_samples)} samples")

    # Remove the sample_type field (it was just for debugging)
    for sample in all_samples:
        del sample["sample_type"]

    return all_samples, existing_ds


def main():
    parser = argparse.ArgumentParser(description="Create and upload interleaved dataset")
    parser.add_argument("--dry-run", action="store_true", help="Create dataset but don't upload")
    args = parser.parse_args()

    # Create the interleaved samples
    interleaved_samples, existing_ds = create_interleaved_dataset()

    # Create the interleaved dataset with matching features
    print("\nCreating HuggingFace Dataset...")

    # Get features from existing dataset to ensure consistency
    existing_features = existing_ds["real_data"].features
    print(f"Using features from existing dataset: {existing_features}")

    interleaved_ds = Dataset.from_list(interleaved_samples)
    # Cast to match existing features
    interleaved_ds = interleaved_ds.cast(existing_features)
    print(f"Interleaved dataset: {interleaved_ds}")
    print(f"Features: {interleaved_ds.features}")

    # Show sample distribution after shuffle
    print("\nFirst 5 samples (showing structure):")
    for i, sample in enumerate(interleaved_samples[:5]):
        num_images = len(sample["images"])
        user_content = sample["messages"][0]["content"]
        content_types = [item["type"] for item in user_content]
        print(f"  {i}: {num_images} image(s), content pattern: {content_types}")

    if args.dry_run:
        print("\n[DRY RUN] Would upload the following dataset:")
        print(f"  - single_image: {len(existing_ds['single_image'])} samples")
        print(f"  - multi_image: {len(existing_ds['multi_image'])} samples")
        print(f"  - real_data: {len(existing_ds['real_data'])} samples")
        print(f"  - interleaved: {len(interleaved_ds)} samples (NEW)")
        return

    # Combine with existing splits
    print("\nCombining with existing splits...")
    combined = DatasetDict({
        "single_image": existing_ds["single_image"],
        "multi_image": existing_ds["multi_image"],
        "real_data": existing_ds["real_data"],
        "interleaved": interleaved_ds,
    })

    print(f"Combined dataset: {combined}")

    # Push to hub
    print(f"\nPushing to HuggingFace Hub: {HF_DATASET}")
    combined.push_to_hub(HF_DATASET)
    print("Done!")

    # Verify upload
    print("\nVerifying upload...")
    verify_ds = load_dataset(HF_DATASET, split="interleaved")
    print(f"Verified interleaved split: {len(verify_ds)} samples")


if __name__ == "__main__":
    main()
