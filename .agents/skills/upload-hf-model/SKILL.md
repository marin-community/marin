---
name: upload-hf-model
description: Upload or publish model weights to the Hugging Face Hub. Use for model repositories, not dataset uploads or Hub downloads.
---

# Upload a model to Hugging Face

- Use the `open-athena` Hugging Face organization when the user has not specified a destination namespace. Honor an explicit destination from the user.
- For every model upload, make sure the model repository's root `README.md` includes this exact text:

  Research artifact. This model has not been properly tested or evaluated and is not necessarily secure. Use at your own risk in production settings.

- Check the published `README.md` after uploading or updating the model to confirm that the text is present.
