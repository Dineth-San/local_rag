import time, tools
# for q in ["What is this document about?", "an exact acronym from your PDF?"]:
#     t = time.time()
#     results = tools.retrieve(q)
#     print(f"\n{q}  ({time.time() - t:.2f}s)")
#     for doc, score in results:
#         print(f"  {score:6.2f}  p.{doc.metadata['page']}  {doc.page_content[:100]!r}")


print(tools.get_document_overview.invoke({}))
print(tools.search_documents.invoke({"query": "your keyword"}))