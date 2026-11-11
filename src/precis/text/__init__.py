"""Paper text: HTML and PDF parsing into ``Document`` records, and passage search.

``precis.text.html`` and ``precis.text.pdf`` turn a downloaded file into a
``Document``; ``precis.text.quality`` decides whether the result is worth
summarizing; ``precis.text.bm25`` indexes a document for the agents'
``search_paper`` tool.
"""
