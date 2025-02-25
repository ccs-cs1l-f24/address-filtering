import csv
import random

# Define the number of rows
num_rows = 100

# Define the starting block number
start_block_number = 16113072

# Define a function to generate random nonzero values
def random_nonzero():
    return random.randint(1, 1000000)  # Adjust the range as needed

# Open a CSV file for writing
with open("transaction_data.csv", mode="w", newline="") as file:
    writer = csv.writer(file)

    # Write the header row
    writer.writerow(['block_numbers',"out_tx","in_tx","out_value","in_value","unique_receivers","unique_senders"])

    # Write the data rows
    for i in range(num_rows):
        writer.writerow([
            start_block_number - i,  # Decrement block number
            random_nonzero(),  # out_tx
            random_nonzero(),  # in_tx
            random_nonzero(),  # out_value
            random_nonzero(),  # in_value
            random_nonzero(),   # unique_receivers
            random_nonzero(),   # unique_senders
        ])

print("CSV file 'transaction_data.csv' generated successfully!")
