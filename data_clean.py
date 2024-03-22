def remove_numbers(input_file, output_file):
    try:
        # Open the input file for reading
        with open(input_file, 'r') as f:
            # Read the content of the file
            content = f.read()
        
        # Remove numbers from the content
        content_without_numbers = ''.join([char for char in content if not char.isdigit() and char != ':'])
        
        # Open the output file for writing
        with open(output_file, 'w') as f:
            # Write the modified content to the output file
            f.write(content_without_numbers)
        
        print("Numbers removed and saved to", output_file)
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print("An error occurred:", e)

# Example usage:
input_file = 'corpus.txt'
output_file = 'corpus_clean.txt'
remove_numbers(input_file, output_file)
