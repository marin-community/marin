---
name: upload-hf-model
description: Upload or publish model weights to the Hugging Face Hub. Use for model repositories, not dataset uploads or Hub downloads.
---

# Upload a model to Hugging Face

- Use the `open-athena` Hugging Face organization when the user has not specified a destination namespace. Honor an explicit destination from the user.
- Upload models under the OpenMDW 1.1 license. Copy the [official license text](https://github.com/OpenMDW/OpenMDW/blob/main/1.1/LICENSE.OpenMDW-1.1) into the model repository's root `LICENSE` file and set `license: openmdw-1.1` in the root `README.md` YAML metadata.
- For every model upload, make sure the model repository's root `README.md` includes this exact text:

  Research artifact. This model has not been properly tested or evaluated and is not necessarily secure. Use at your own risk in production settings.

- For every model upload or update, also include this exact responsible-use paragraph in the model repository's root `README.md`. Preserve the research-artifact warning above; do not add duplicate copies of either paragraph:

  We release our tools to the public with the intent that they are used safely, ethically, and responsibly. They are meant to contribute to the public's understanding of science, and aid in its advancement. By using this Marin model, you agree to comply with all applicable law.

- Before publishing, check the local model card for both required paragraphs and the OpenMDW 1.1 license metadata. Do not upload a card that omits them.
- Check the published `README.md` and `LICENSE` after uploading or updating the model to confirm that the research-artifact warning, responsible-use paragraph, OpenMDW 1.1 license metadata, and full license text are present.
