def translate_text(text, model, tokenizer, device, max_length=5000):
    """
    Generates a summary for the given text using a pre-trained model.

    Args:
        text (str): The text to be summarized.
        max_length (int): The maximum length of the input text for the model.

    Returns:
        str: The generated summary of the input text.
    """
    # Encode the input text using the tokenizer.
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=max_length, truncation=False)

    # Move the encoded text to the same device as the model (e.g., GPU or CPU)
    inputs = inputs.to(device)

    # Generate translate IDs with the model. num_beams controls the beam search width.
    # early_stopping is set to False for a thorough search, though it can be set to True for faster results.
    translate_ids = model.generate(inputs, max_length=2000, num_beams=30, early_stopping=False)

    # Decode the generated IDs back to text, skipping special tokens like padding or EOS.
    translation = tokenizer.decode(translate_ids[0], skip_special_tokens=True)

    # Return the generated translation
    return translation