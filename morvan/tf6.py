#https://www.youtube.com/watch?v=hWmbAglaHxk&index=6&list=PLXO45tsB95cJHXaDKpbwr5fC_CCYylw1f
#https://www.youtube.com/watch?v=Tve-Huc1IRo  (matrix multiplication)
import tensorflow as tf

#  1 row with 2 columns
matrix1 = tf.constant([[3,3]])

#  2 rows with 1 column
matrix2 = tf.constant([[2], [2]])

#matrix multiply
#if matrix is 2x3 (that means 2 rows and 3 columns)
# if matrix1 is 2x3 and matrix2 is 3x4 they will equal a matrix of size 2x4 (the first matrix' row count and the second matrix' column count)
# number of columns in matrix1 must equal number of rows in matrix2 in order to multiply otherwise you can't multiply them

product = tf.matmul(matrix1, matrix2)   # matrix multiply can also be done with numpy library as np.dot(matrix1, matrix2)
print('product', product)


#method 1 to use session

sess = tf.Session()
result = sess.run(product)
print('result', result)
sess.close() #closes session, but session auto closes so not necessary

#method2 to use session