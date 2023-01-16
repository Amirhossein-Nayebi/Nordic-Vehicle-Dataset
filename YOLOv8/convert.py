# Open the input file for reading
with open('./TestYOLOV8/labels.txt', 'r') as input_file:
    # Open the output file for writing
    with open('./TestYOLOV8/output.txt', 'w') as output_file:
        # Iterate over each line in the input file
        for line in input_file:
            line = line.strip() + "\n"
            # Write the line to the output file
            output_file.write(line)
