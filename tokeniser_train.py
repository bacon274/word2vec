import sentencepiece as spm

input_file: 'corpus_clean.txt'
output_file: 'tokens.txt'


# Train SentencePiece model
spm.SentencePieceTrainer.train('--input=corpus_clean.txt --model_prefix=tokenizer --vocab_size=1000')

