import sqlite3


def insert_data(sentence, label, joining):
    # connecting to the database
    connection = sqlite3.connect("db/myTable.db")

    # cursor
    crsr = connection.cursor()

    # SQL command to create a table in the database
    sql_command = """CREATE TABLE TOPICS (  
        id INTEGER PRIMARY KEY AUTOINCREMENT,  
        sentence TEXT,  
        label INTEGER,  
        joining DATE,
        is_check INTEGER
        );"""

    # execute the statement

    try:
        crsr.execute(sql_command)
    except:
        pass
    sql_command = f"""INSERT INTO TOPICS(sentence, label, joining, is_check) VALUES (?, ?, ?, ?);"""
    crsr.execute(sql_command,  (sentence, label, joining, 0))
    connection.commit()

    # close the connection
    connection.close()
    return "save to db successfull!"

# def

# # SQL command to insert the data in the table
# sql_command = """INSERT INTO TOPICS(sentence, label, joining) VALUES ("Bansal", "M", "2014-03-28");"""
# crsr.execute(sql_command)

# # another SQL command to insert the data in the table
# sql_command = """INSERT INTO TOPICS(sentence, label, joining) VALUES ("Gates", "M", "1980-10-28");"""
# crsr.execute(sql_command)

# # To save the changes in the files. Never skip this.
# If we skip this, nothing will be saved in the database.
