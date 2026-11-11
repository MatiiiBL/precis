"""Where papers come from: the arXiv API and arxiv.org downloads, or a local directory.

``precis.sources.arxiv`` searches the API and fetches metadata by id,
``precis.sources.fulltext`` downloads and parses full text (HTML first, PDF as
the fallback), and ``precis.sources.local`` reads a directory of PDFs. All
arXiv traffic goes through one ``precis.sources.http.ArxivHttp``.
"""
