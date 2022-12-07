from fastapi import FastAPI, File, UploadFile
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import towhee
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import towhee
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from pydantic import BaseModel
from sqlalchemy.orm import Session
from fastapi import Depends, FastAPI, HTTPException
import csv
import codecs

SQLALCHEMY_DATABASE_URL = "postgresql://minhson:test@127.0.0.1:5431/question"
# connect milvus

connections.connect(host='127.0.0.1', port='19530')


# connect postgres
engine = create_engine(
    SQLALCHEMY_DATABASE_URL
)


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class Ques_Ans(Base):
    __tablename__ = "chatbot_eng"
    id = Column(Integer, primary_key=True, index=True)
    question = Column(String, index=True)
    answer = Column(String, index=True)

# model


class Ques_Ans_Schema(BaseModel):
    id: int
    ques: str
    ans: str

# repositories


def get_ques(db: Session, id: int):
    return db.query(Ques_Ans).filter(Ques_Ans.id == id).first()


def get_last_id_in_db():
    try:
        obj = session.query(Ques_Ans).order_by(Ques_Ans.id.desc()).first()
        return int(obj.id) + 1
    except:
        return 1


def get_ans(db: Session, res, question):
    list_ans = []
    for entity in res:
        list_ans.append((db.query(Ques_Ans).filter(
            Ques_Ans.id == entity.id).first()).answer)
        if entity.score > 0.5:
            with open('question.csv', 'a') as f:
                row = [-1, question]
                writer = csv.writer(f)
                writer.writerow(row)
    return list_ans


def write_ques_ans(ques_ans: Ques_Ans_Schema):
    db_ques_ans = Ques_Ans(
        id=ques_ans.id, question=ques_ans.ques, answer=ques_ans.ans)
    session.add(db_ques_ans)
    session.commit()
    session.refresh(db_ques_ans)
    return db_ques_ans

# hi
# service


def upload_csv_to_db(file):
    csvReader = csv.DictReader(codecs.iterdecode(file.file, 'utf-8'))
    new_id = get_last_id_in_db()
    for rows in csvReader:
        print(rows)
        ques_ans = Ques_Ans_Schema(
            id=new_id, ques=rows['question'], ans=rows['answer'])
        write_ques_ans(ques_ans)
        new_id += 1

# Dependency


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


session = next(get_db())

#
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/message/{question}")
async def ask_and_answer(question):
    try:
        dc = (towhee.dc([question])
              .text_embedding.dpr(model_name="facebook/dpr-ctx_encoder-single-nq-base")
              .runas_op(func=lambda x: x.squeeze(0))
              .tensor_normalize()
              .milvus_search(collection='question_answer', limit=1)
              .runas_op(lambda res: get_ans(session, res, question))
              .to_list()
              )
        return dc[0]
    except:
        print("error")


@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    upload_csv_to_db(file)

# get file csv from database Postgress and add file csv to Milvus
@app.get('/get_file')
def all(db: Session = Depends(get_db)):
    lines = db.query(Ques_Ans).all()
    ans = []
    for line in lines:

        header = ['id', 'question', 'answer']
        line_data = [line.id, line.question, line.answer]
        ans.append(line_data)

    with open('question_answer.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(header)

        writer.writerows(ans)

    def create_milvus_collection(collection_name, dim):
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)

        fields = [
            FieldSchema(name='id', dtype=DataType.INT64,
                        descrition='ids', is_primary=True, auto_id=False),
            FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR,
                        descrition='embedding vectors', dim=dim)
        ]
        schema = CollectionSchema(
            fields=fields, description='reverse image search')
        collection = Collection(name=collection_name, schema=schema)

        # create IVF_FLAT index for collection.
        index_params = {
            'metric_type': 'L2',
            'index_type': "IVF_FLAT",
            'params': {"nlist": 2048}
        }
        collection.create_index(field_name="embedding",
                                index_params=index_params)
        return collection

    collection = create_milvus_collection('question_answer', 768)

    dc = (
        towhee.read_csv('question_answer.csv')
        .runas_op['id', 'id'](func=lambda x: int(x))
        .text_embedding.dpr['question', 'vec'](model_name="facebook/dpr-ctx_encoder-single-nq-base")
        .runas_op['vec', 'vec'](func=lambda x: x.squeeze(0))
        .tensor_normalize['vec', 'vec']()
        .to_milvus['id', 'vec'](collection=collection, batch=100)
    )

    towhee.read_csv('question_answer.csv').show()

    dc1 = towhee.read_csv('question_answer.csv').head(2).to_list()
    dc2 = towhee.read_csv('question_answer.csv').runas_op['id', 'id'](
        func=lambda x: int(x)).head(2).to_list()

    towhee.read_csv('question_answer.csv').head(2).text_embedding.dpr['question', 'vec'](
        model_name="facebook/dpr-ctx_encoder-single-nq-base").show()

    return lines
