import mysql.connector
import csv
import os

db_config = {
    'host': 'localhost',       # or your database host
    'user': 'your_username',
    'password': 'your_password',
    'database': 'your_database'
}

table_name = 'your_table_name'

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
output_file = os.path.join(desktop_path, f"{table_name}_data.csv")

try:
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()

    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()

    column_names = [desc[0] for desc in cursor.description]

    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(column_names)  # write headers
        writer.writerows(rows)        # write data rows

    print(f"Data has been successfully saved to {output_file}")

except mysql.connector.Error as err:
    print(f"Error: {err}")

finally:
    if connection.is_connected():
        cursor.close()
        connection.close()
