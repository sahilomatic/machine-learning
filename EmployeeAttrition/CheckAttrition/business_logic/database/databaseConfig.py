from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
class CreateConnection:
    def connect(self):
        engine = create_engine('postgresql://postgres:postgres@postgresinstance.crgqhdyqw8yy.us-east-2.rds.amazonaws.com:5432/postgres')    
        
        Session = sessionmaker(bind=engine)
        
        session = Session()
        
        
        return session
    
    
if __name__ == "__main__":
    dao =  CreateConnection()
    dao.connect()    

