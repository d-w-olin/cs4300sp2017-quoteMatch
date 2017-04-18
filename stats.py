import csv

class Quote:
	""" a simple quote instance """
	def __init__(self, qid, author, quote):
		assert type(qid) == int
		assert type(author) == str
		assert type(quote) == str

		self.id = qid
		self.author = author
		self.quote = quote

	def __repr__(self):
		return '<quote object '+str(self.id)+'>'

	def __str__(self): return self.quote

def parse_quotes(f):
	quotes = set()
	reader = csv.reader(f, delimiter=',')
	for qid,author,quote in reader:
		quotes.add(Quote(int(qid), author, quote))
	return quotes

parse_quotes(open('/Users/galenweld/GoogleDrive/School/CS4300/project/quotes.csv'))