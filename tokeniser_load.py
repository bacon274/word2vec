import sentencepiece as spm

# Load trained model
sp = spm.SentencePieceProcessor()
sp.load('tokenizer.model')

# methods_attributes = dir(sp)
# print(methods_attributes)


def encode(input_text):
    return sp.encode_as_ids(input_text)

def decode(tokens):
    return sp.decode_pieces(tokens)



# # # Tokenize text
# text = "Hello, how are you? This is a test."
# tokenized_text = encode(text)
# print("Tokenized text:", tokenized_text)

# decoded_text = decode(tokenized_text)
# print("Decoded text:", decoded_text)


