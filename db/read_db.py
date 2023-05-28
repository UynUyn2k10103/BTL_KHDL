# importing the module
import sqlite3

# connect withe the myTable database
connection = sqlite3.connect("db/myTable.db")

# cursor object
crsr = connection.cursor()

crsr.execute("SELECT * FROM TOPICS")

ans = crsr.fetchall()

for row in ans:
    print(row)
