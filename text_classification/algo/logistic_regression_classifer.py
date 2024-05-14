from neo4j import GraphDatabase

def connect_to_neo4j_aura():
    """Connect to a Neo4j Aura database using environment variables for security."""
    uri = "neo4j+s://df0f8145.databases.neo4j.io" #os.getenv("NEO4J_URI")
    user = "neo4j" #os.getenv("NEO4J_USER")
    password = "67rvGvKtZY_h2xK1XIB5HUz-Qum5hO24d478QRpc0JI" #os.getenv("NEO4J_PASSWORD")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    return driver

def run_query(driver, query):
    """Execute a Cypher query and return a list of results."""
    with driver.session() as session:
        result = session.run(query)
        return [record for record in result]

def insert_record(driver, label, name):
    """Insert a record into the Neo4j database."""
    query = f"CREATE (n:{label} {{name: $name}})"
    with driver.session() as session:
        session.run(query, name=name)

def main():
    try:
        # Connect to Neo4j Aura
        driver = connect_to_neo4j_aura()

        # Example query
        query = "MATCH (n) RETURN COUNT(n) AS node_count"
        records = run_query(driver, query)

        # Print results
        for record in records:
            print(record["node_count"])

        # Insert a dummy record
        insert_record(driver, "Dummy 2", "Dummy Record 222")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the driver
        driver.close()

if __name__ == "__main__":
    main()
