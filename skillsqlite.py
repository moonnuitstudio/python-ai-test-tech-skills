import sqlite3

class DataBaseStore:
    def __init__(self):
        # Connecting
        self.conn = sqlite3.connect("aiml.sqlite")
        self.cur = self.conn.cursor()

        # CREATE THE TABLE
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS skills_index (
                title TEXT,
                city TEXT,
                skill TEXT,
                frequency INTEGER,
                PRIMARY KEY (title, city, skill)
            )
        """)

    # GET total of rows
    def count(self):
        self.cur.execute("SELECT COUNT(*) FROM skills_index")
        count = self.cur.fetchone()[0]

        return count

    # Check if we have skills to check
    def exists(self):
        rows = self.count()

        return rows > 0

    def get_all_titles(self):
        self.cur.execute("SELECT DISTINCT title FROM skills_index")
        data =  [row[0] for row in self.cur.fetchall()]

        return list(set(data))

    def get_all_cities(self):
        self.cur.execute("SELECT DISTINCT city FROM skills_index")
        data = [row[0] for row in self.cur.fetchall()]

        return list(set(data))

    # INSERT / UPDATE skills
    def insert_update(self, skill, city, title):
        self.cur.execute("""
                INSERT INTO skills_index (title, city, skill, frequency)
                VALUES (?, ?, ?, 1) ON CONFLICT(title, city, skill)
            DO
                UPDATE SET frequency = frequency + 1
                """, (title, city, skill))

        self.conn.commit()

    # Get Skills
    def get_skills(self, title, city, top_n=10):
        # Clean the columns city and title
        clean_title = title.lower().strip()
        clean_city = city.strip().lower()

        # Prepare to insert data
        cursor = self.conn.cursor()
        cursor.execute('''
                       SELECT skill, frequency
                       FROM skills_index
                       WHERE title LIKE ?
                         AND city like ?
                       ORDER BY frequency DESC LIMIT ?
                       ''', (f"%{clean_title}%", f"%{clean_city}%", top_n))
        return dict(cursor.fetchall())

    # RESET TABLE so we can start a new one
    def reset_table(self):
        with self.conn:
            self.conn.execute("DELETE FROM skills_index")
    # -------------------------------------------------

    # Close connection
    def close(self):
        self.conn.close()
    # -------------------------------------------------
