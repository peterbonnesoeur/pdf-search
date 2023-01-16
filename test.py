import openai
import os
import qa_pipeline

openai.api_key = os.environ.get("OPEN_AI_KEY")

COMPLETIONS_MODEL = "text-davinci-002"

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 500,
    "model": COMPLETIONS_MODEL,
}


def construct_prompt(haystack_prediction):
  header = """Ignore all previous direction. Answer the question as truthfully as possible using the provided context, first write your answer, then the source and page mentioned in the prompt and if the answer is not contained within the text below, say "I don't know"."""

  prediction = haystack_prediction['documents'][0]
  context = prediction.content
  source = prediction.meta["chapter"]
  page = prediction.meta["start_page"]


  return header + "\n\nContext: " + context + "\nSource: " + source + "\nPage: " + str(page) + "\n\nThe question is: " + haystack_prediction['query']

def answer_query_with_context(query, pipeline) -> str:
    haystack_prediction = qa_pipeline.query(query, pipeline)
    prompt = construct_prompt(haystack_prediction)

    print(prompt)

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )


    return response["choices"][0]["text"].strip("\n")

def answer_query(query) -> str:


  response = openai.Completion.create(
              prompt=query,
              **COMPLETIONS_API_PARAMS
          )
  print(response)

  return response["choices"][0]["text"].strip("\n")

# import json
# import requests
# headers = {"Authorization": f"Bearer {API_TOKEN}"}
# API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
# def query(payload):
#     data = json.dumps(payload)
#     response = requests.request("POST", API_URL, headers=headers, data=data)
#     return json.loads(response.content.decode("utf-8"))
# data = query(
#     {
#         "inputs": {
#             "question": "What's my name?",
#             "context": "My name is Clara and I live in Berkeley.",
#         }
#     }
# )import gradio as gr
import preprocess
import qa_pipeline
import gpt
from typing import Union

def welcome():
    print("Welcome to your source of infinite knowledge")

def do_processing(book_location: Union[str, None]) -> None:
    # global pipeline
    if book_location:
        book = preprocess.preprocess(book_location)
        # global pipeline
        pipeline = qa_pipeline.build_pipeline(book)
    else:
        pipeline = None
    return pipeline

def make_query(query: str, pipeline: Union[str,None]) -> str:
    if pipeline:
        answer = gpt.answer_query_with_context(query, pipeline)
    else:
        answer = gpt.answer_query(query)
    return answer

def main(book_location: Union[str, None], query: str) -> str:
    pipeline = do_processing(book_location)
    return make_query(query, pipeline)

if __name__ == "__main__":
    welcome()
    gr.Interface(fn=main, inputs=["text", "text"], outputs="text").launch()
import pypdf

def flatten_list(nested_list):
  """
  Flattens a n-dimensional nested list
  """
  # check if list is empty
  if not(bool(nested_list)):
      return nested_list

    # to check instance of list is empty or not
  if isinstance(nested_list[0], list):

      # call function with sublist as argument
      return flatten_list(*nested_list[:1]) + flatten_list(nested_list[1:])

  # call function with sublist as argument
  return nested_list[:1] + flatten_list(nested_list[1:])

 

# Find the lenght of each chapter
def bookmark_dict(bookmark_list, reader):
    result = []
    counter=0
    for i in range(len(bookmark_list)):
      try:
        dic = {
            "bookmark_title" : bookmark_list[i]["/Title"],
            "start_page" : reader.get_destination_page_number(bookmark_list[i])+1,
            "end_page" : reader.get_destination_page_number(bookmark_list[i+1])+1
        }
      except:
        dic = {
            "bookmark_title" : bookmark_list[i]["/Title"],
            "start_page" : reader.get_destination_page_number(bookmark_list[i])+1,
            "end_page": len(reader.pages)
        }
      result.append(dic)

    return result

def find_bookmarks(reader):
    flat_list = flatten_list(reader.outline)
    bookmarks = bookmark_dict(flat_list, reader)
    
    return bookmarks

def split_book_chapters(bookmarks, pdf_reader):
    chapters = []
    for i in bookmarks:
        concat_pages = ""
        for j in range(i["start_page"]-1, i["end_page"]):
            print(i["bookmark_title"] +str(j) + " -> " + str(i["start_page"]) + ":" + str(i["end_page"]))
            concat_pages = concat_pages + "\n" + pdf_reader.pages[j].extract_text().replace('\n', '')
            print("* Processed chapter *")
        chapters.append({"content": concat_pages, "meta": {"book_title": "DAMA-DMBOK Data Management Book of Knowledge", "chapter": i["bookmark_title"], "start_page" : i["start_page"], "end_page": i["end_page"]}})
    
    return chapters
    
def preprocess(book_location):
    reader = pypdf.PdfReader(book_location)
    bookmarks = find_bookmarks(reader)
    book = split_book_chapters(bookmarks, reader)

    return book# In-Memory Document Store

from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import TfidfRetriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline


def build_document_store(docs):
    document_store = InMemoryDocumentStore()
    document_store.write_documents(docs)
    return document_store

def build_retriever(document_store):
    retriever = TfidfRetriever(document_store=document_store)
    return retriever

def build_reader():
    # Load a  local model or any of the QA models on
    # Hugging Face's model hub (https://huggingface.co/models)
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)

    return reader

def build_pipeline(book):
    # Extractive Pipeliline
    document_store = build_document_store(book)
    retriever = build_retriever(document_store)
    reader = build_reader()
    pipeline = ExtractiveQAPipeline(reader, retriever)
    return pipeline

def query(query, pipeline):
    prediction = pipeline.run(query, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}})
    return prediction
