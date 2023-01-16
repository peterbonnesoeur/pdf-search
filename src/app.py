import gradio as gr
import preprocess
import qa_pipeline
import gpt
from typing import Union
from haystack.pipelines import ExtractiveQAPipeline

def welcome()-> None:
    """
    A simple welcome message.
    """
    print("Welcome to your source of infinite knowledge")

def do_processing(book_location: Union[str, None]) -> None:
    """
    Preprocess the book and build the pipeline.
    :param book_location: The location of the book.
    :return: None
    """
    if book_location:
        book = preprocess.preprocess(book_location)
        pipeline = qa_pipeline.build_pipeline(book)
    else:
        pipeline = None
    return pipeline

def make_query(query: str, pipeline: Union[None,ExtractiveQAPipeline]) -> str:
    """
    answer the question based on the context if exists or not
    :param query: question to be answered
    :param pipeline: context of the question if exists
    :return: answer
    """
    if pipeline:
        answer = gpt.answer_query_with_context(query, pipeline)
    else:
        answer = gpt.answer_query(query)
    return answer

def main(book_location: Union[str, None], query: str) -> str:
    """
    main function to call the other functions and return the answer
    :param book_location: location of the book
    :param query: question to be answered
    :return: answer
    """
    pipeline = do_processing(book_location)
    return make_query(query, pipeline)

if __name__ == "__main__":
    welcome()
gr.Interface(fn=main, inputs=[
    gr.Textbox(lines=5, label="Book Location"),
    gr.Textbox(lines=5, label="Question"),
    ] , outputs="text").launch()
