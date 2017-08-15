# -*- coding: UTF-8 -*-

# py2
# import pymysql

# py3
import pymysql

class ConnMysql:
	def __init__(self, host, port, user, password, db):
		self.host = host
		self.port = port
		self.user = user
		self.password = password
		self.db = db

	def connectMysql(self):
		try:
			self.conn = pymysql.connect(host=self.host, port=self.port, user=self.user, passwd=self.password, db=self.db, charset='utf8')
			self.cursor = self.conn.cursor()
			print("connect successful")
		except:
			print("connect mysql error.")

	def queryData(self, sql):
		"""
		执行sql语句
		:param sql: 
		:return: 返回元组tuple类型
		"""
		try:
			self.cursor.execute(sql)
			return self.cursor.fetchall()
		except:
			print("query execute failed")
			return None

	def insertData(self, sql):
		try:
			self.cursor.execute(sql)
		except:
			print("insert execute failed")

	def insertManyData(self, sql, list):
		try:
			self.cursor.executemany()
		except:
			print "exceute many failed"

	def updateData(self, sql):
		try:
			self.cursor.execute(sql)
			self.conn.commit()
		except:
			self.conn.rollback()

	def closeMysql(self):
		self.cursor.close()
		self.conn.close()