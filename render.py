from re import X

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from image import Color, Image
from matplotlib.pyplot import angle_spectrum
from model import Model
from shape import Line, Point, Triangle
from vector import Vector
import sys

width = 500
height = 300

# Load the model
model = Model('data/headset.obj')
model.normalizeGeometry()


def getOrthographicProjection(x, y, z):
	# Convert vertex from world space to screen space
	# by dropping the z-coordinate (Orthographic projection)

	screenX = int((x+1.0)*width/2.0)
	screenY = int((y+1.0)*height/2.0)

	return screenX, screenY


def updateData(df):

	for i, j in df.iterrows():
		
		df.at[i, ' gyroscope.X'] = j[' gyroscope.X'] * (np.pi / 180)
		df.at[i, ' gyroscope.Y'] = j[' gyroscope.Y'] * (np.pi / 180)
		df.at[i, ' gyroscope.Z'] = j[' gyroscope.Z'] * (np.pi / 180)

		a_values = [j[' accelerometer.X'],j[' accelerometer.Y'],j[' accelerometer.Z']]
		if np.isnan in a_values:
			a_mag = 0
		else:
			a_mag = np.sqrt(sum(x**2 for x in a_values))
		if a_mag != 0:
			df.at[i, ' accelerometer.X'] = j[' accelerometer.X'] / a_mag
			df.at[i, ' accelerometer.Y'] = j[' accelerometer.Y'] / a_mag
			df.at[i, ' accelerometer.Z'] = j[' accelerometer.Z'] / a_mag
		
		m_values = [j[' magnetometer.X'],j[' magnetometer.Y'],j[' magnetometer.Z ']]
		if np.isnan in m_values:
			m_mag = 0
		else:
			m_mag = np.sqrt(sum([x**2 for x in m_values]))
		if m_mag != 0:
			df.at[i, ' magnetometer.X'] = j[' magnetometer.X'] / m_mag
			df.at[i, ' magnetometer.Y'] = j[' magnetometer.Y'] / m_mag
			df.at[i, ' magnetometer.Z '] = j[' magnetometer.Z '] / m_mag

	return df


def rotate(angle, type):

	angle = angle * (np.pi/180)

	c = np.cos(angle)
	s = np.sin(angle)

	if type == 1:
		rotation_matrix = np.matrix([
			[1,0,0,0],
			[0,c,-s,0],
			[0,s,c,0],
			[0,0,0,1]
		])

	if type == 2:
		rotation_matrix = np.matrix([
			[c,0,s,0],
			[0,1,0,0],
			[-s,0,c,0],
			[0,0,0,1]
		])

	if type == 3:
		rotation_matrix = np.matrix([
			[c,-s,0,0],
			[s,c,0,0],
			[0,0,1,0],
			[0,0,0,1]
		])

	return rotation_matrix


def scale(x, y, z):
	scale_matrix = np.matrix([
		[x,0,0,0],
		[0,y,0,0],
		[0,0,z,0],
		[0,0,0,1]
	])
	return scale_matrix


def translate(x, y, z):
	translate_matrix = np.matrix([
		[1,0,0,x],
		[0,1,0,y],
		[0,0,1,z],
		[0,0,0,1]
	])
	return translate_matrix


def normalise(vector):
	mag = np.sqrt(np.sum(vector**2))
	if mag == 0:
		return vector
	else:
		return vector / mag


def view():

	eye_pos = np.array([0,0,-2.5])
	ref_pos = np.array([0,1,1])
	up_vec = np.array([0,1,-2.5])

	up_vec = normalise(up_vec)

	z_temp = eye_pos - ref_pos
	z = normalise(z_temp)

	x_temp = np.cross(up_vec, z)
	x = normalise(x_temp)

	y = np.cross(z, x)

	R = np.matrix([
		[x[0],x[1],x[2],0],
		[y[0],y[1],y[2],0],
		[z[0],z[1],z[2],0],
		[0,0,0,1]		
	])

	T = np.matrix([
		[1,0,0,-(eye_pos[0])],
		[0,1,0,-(eye_pos[1])],
		[0,0,1,-(eye_pos[2])],
		[0,0,0,1]
	])

	view_matrix = np.dot(R,T)

	return view_matrix
'''
	view_matrix = np.matrix([
		[x[0],x[1],x[2],-((x[0]*eye_pos[0]) + (x[1]*eye_pos[1]) + (x[2]*eye_pos[2]))],
		[y[0],y[1],y[2],-((y[0]*eye_pos[0]) + (y[1]*eye_pos[1]) + (y[2]*eye_pos[2]))],
		[z[0],z[1],z[2],-((z[0]*eye_pos[0]) + (z[1]*eye_pos[1]) + (z[2]*eye_pos[2]))],
		[0,0,0,1]		
	])
'''


def getPerspectiveProjection3(x, y, z):

	fovy = 45
	aspect = width / height

	l = -(width/2)
	t = height/2
	f = 750
	r = width/2
	b = -(height/2)
	n = 200

	fovy_temp = 1/np.tan(fovy/2)

	S = 1 / np.tan((fovy/2) * (np.pi/180))

	symmetric_projection = np.matrix([
		[fovy_temp/aspect,0,0,0],
		[0,fovy_temp,0,0],
		[0,0,-((f+n)/(f-n)),-((2*f*n)/(f-n))],
		[0,0,-1,0]
	])

	asymmetric_projection = np.matrix([
		[((2*n)/(r-l)),0,((r+l)/(r-l)),0],
		[0,((2*n)/(t-b)),((t+b)/(t-b)),0],
		[0,0,-((f+n)/(f-n)),-((2*f*n)/(f-n))],
		[0,0,-1,0]
	])

	view_matrix = view()

	vector = np.array([[x],[y],[z],[1]])

	clip = asymmetric_projection * view_matrix * vector

	new_x = clip[0,0]/clip[3,0]
	new_y = clip[1,0]/clip[3,0]
	new_z = clip[2,0]/clip[3,0]

#	first = asymmetric_projection * view_matrix
#	result = (vector * first).tolist()[0]
#
#	new_x = result[0]/result[3]
#	new_y = result[1]/result[3]
#	new_z = result[2]/result[3]

	screenX = int((new_x+1.0)*(width/2.0))
	screenY = int((new_y+1.0)*(height/2.0))
	screenZ = int(new_z*(1.0/2.0))

	return screenX, screenY, screenZ	


def axisToQuart(angle, points):

	a = np.cos(angle/2)
	
	b = points[0] * np.sin(angle/2)

	c = points[1] * np.sin(angle/2)

	d = points[2] * np.sin(angle/2)

	quartonian = [a,b,c,d]

	return quartonian


def quartToAxis(quartonian):

	if quartonian[0] == 1:
		angle = 0
		x = 1
		y = 0
		z = 0
	else:

		a = quartonian[0]
		b = quartonian[1]
		c = quartonian[2]
		d = quartonian[3]

		angle = 2 * np.arccos(a)

		x = b / np.sin(angle/2)
		y = c / np.sin(angle/2)
		z = d / np.sin(angle/2)

	points = [x,y,z]

	return angle, points


def eulerToQuart(roll, pitch, yaw):

	a = (np.cos(roll/2)*np.cos(pitch/2)*np.cos(yaw/2)) + (np.sin(roll/2)*np.sin(pitch/2)*np.sin(yaw/2))

	b = (np.sin(roll/2)*np.cos(pitch/2)*np.cos(yaw/2)) - (np.cos(roll/2)*np.sin(pitch/2)*np.sin(yaw/2))

	c = (np.cos(roll/2)*np.sin(pitch/2)*np.cos(yaw/2)) + (np.sin(roll/2)*np.cos(pitch/2)*np.sin(yaw/2))

	d = (np.cos(roll/2)*np.cos(pitch/2)*np.sin(yaw/2)) - (np.sin(roll/2)*np.sin(pitch/2)*np.cos(yaw/2))

	quartonian = [a, b, c, d]

	return quartonian


def quartToEuler(quartonian):

	a = quartonian[0]
	b = quartonian[1]
	c = quartonian[2]
	d = quartonian[3]

	roll = np.arctan((2*((a*b) + (c*d))) / ((a**2) - (b**2) - (c**2) + (d**2)))

	pitch = np.arcsin(2*((a*c) - (b*d)))

	yaw = np.arctan((2*((a*d) + (b*c))) / ((a**2) + (b**2) - (c**2) - (d**2)))

	return roll, pitch, yaw


def quartInverse(quartonian):

	a = quartonian[0]
	b = -(quartonian[1])
	c = -(quartonian[2])
	d = -(quartonian[3])

	inverse = [a, b, c, d]

	return inverse


def quartProduct(x, y):

	a = (x[0]*y[0]) - (x[1]*y[1]) - (x[2]*y[2]) - (x[3]*y[3])
	b = (x[0]*y[1]) + (x[1]*y[0]) - (x[2]*y[3]) + (x[3]*y[2])
	c = (x[0]*y[2]) + (x[1]*y[3]) + (x[2]*y[0]) - (x[3]*y[1])
	d = (x[0]*y[3]) - (x[1]*y[2]) + (x[2]*y[1]) + (x[3]*y[0])

	product = [a, b, c, d]

	return product


def deadReckoning(gyro, prev):

	l = np.sqrt(sum([x**2 for x in gyro]))

	angle = l * (100/256)

	v = [x/l for x in gyro]

	quartonian = axisToQuart(angle, v)

	current_orientation = quartProduct(prev, quartonian)

	return current_orientation


def quartToMatrix(quartonian):

	a = quartonian[0]
	b = quartonian[1]
	c = quartonian[2]
	d = quartonian[3]

	rotation_matrix = np.matrix([
		[(2*((a**2) + (b**2)))-1, 2*((b*c) - (a*d)), 2*((b*d) + (a*c))],
		[2*((b*c) + (a*d)), (2*((a**2) + (c**2)))-1, 2*((c*d) - (a*b))],
		[2*((b*d) - (a*c)), 2*((c*d) + (a*b)), (2*((a**2) + (d**2)))-1]
	])

	return rotation_matrix


def accelerometer(accel_values, current_orientation):

	inverse_orientation = quartInverse(current_orientation)

	accel_quartonian = [0, accel_values[0], accel_values[1], accel_values[2]]
	accel_temp = quartProduct(current_orientation, accel_quartonian)
	accel_quartonian = quartProduct(accel_temp, inverse_orientation)

	_, accel_world = quartToAxis(accel_quartonian)
	accel_mag = np.sqrt(sum(x**2 for x in accel_world))
	accel_norm = [x/accel_mag for x in accel_world]

	phi = np.arccos(accel_norm[1])

	n = [-accel_norm[2], 0, -accel_norm[0]]
	n_mag = np.sqrt(sum(x**2 for x in n))
	n_norm = [x/n_mag for x in n]

	alpha = 0.5

	final_temp = axisToQuart((1-alpha)*phi, n_norm)

	gyro_accel = quartProduct(final_temp, current_orientation)

	gyro_accel_inverse = quartInverse(gyro_accel)

	return gyro_accel, gyro_accel_inverse


def angularDifference(reference, actual):

	angle1 = np.arctan2(actual[0], actual[2])

	angle2 = np.arctan2(reference[0], reference[2])

	error = abs(angle1 - angle2)

	return angle1, angle2, error


def getVertexNormal(vertIndex, faceNormalsByVertex):
	# Compute vertex normals by averaging the normals of adjacent faces
	normal = Vector(0, 0, 0)
	for adjNormal in faceNormalsByVertex[vertIndex]:
		normal = normal + adjNormal

	return normal / len(faceNormalsByVertex[vertIndex])


def output(df, proj, type):

	count = 0

	current_orientation = [1,0,0,0]

	plt.ion()

	if type == 'mag':

		actual = [0,1,0]

		errors = []

	for i, j in df.iterrows():

		count += 1

		image = Image(width, height, Color(255, 255, 255, 255))
		zBuffer = [-float('inf')] * width * height


		gyro = [j[' gyroscope.X'], j[' gyroscope.Y'], j[' gyroscope.Z']]

		current_orientation = deadReckoning(gyro, current_orientation)

		inverse_prev = quartInverse(current_orientation)


		if type == 'accel':

			accel_values = [j[' accelerometer.X'], j[' accelerometer.Y'], j[' accelerometer.Z']]

			gyro_accel, gyro_accel_inverse = accelerometer(accel_values, current_orientation)
		

		elif type == 'mag':
			
			accel_values = [j[' magnetometer.X'], j[' magnetometer.Y'], j[' magnetometer.Z ']]

			reference, actual = actual, accel_values

			gyro_accel, gyro_accel_inverse = accelerometer(accel_values, current_orientation)

			alpha_2 = 0.001

			angle1, angle2, error = angularDifference(reference, actual)

			errors.append(error)

			mag_quartonian = axisToQuart(-alpha_2*(angle1 - angle2), [0,1,0])

			mag_orientation = quartProduct(mag_quartonian, current_orientation)

			mag_inverse = quartInverse(mag_orientation)


		for face in model.faces:
			p0, p1, p2 = [model.vertices[i] for i in face]
			n0, n1, n2 = [vertexNormals[i] for i in face]

		# Define the light direction
			lightDir = Vector(0, 0, -1)

		# Set to true if face should be culled
			cull = False

		# Transform vertices and calculate lighting intensity per vertex
			transformedPoints = []

			for p, n in zip([p0, p1, p2], [n0, n1, n2]):


				point = [0, p.x, p.y, p.z]


				if type == 'accel':
					first_mult = quartProduct(gyro_accel, point)
					second_mult = quartProduct(first_mult, gyro_accel_inverse)

				elif type == 'mag':
					first_mult = quartProduct(mag_orientation, point)
					second_mult = quartProduct(first_mult, mag_inverse)

				else:
					first_mult = quartProduct(current_orientation, point)
					second_mult = quartProduct(first_mult, inverse_prev)


				_, new_point = quartToAxis(second_mult)

				rotation_matrix = quartToMatrix(second_mult)

				view_matrix = view()

				new_n = np.transpose(np.linalg.inv(rotation_matrix * view_matrix[:3,:3])) * np.array([[n.x], [n.y], [n.z]])

				new_n = new_n.tolist()


				if proj == 'pers':
					screenX, screenY, screenZ = getPerspectiveProjection3(new_point[0], new_point[1], new_point[2])
					#intesity = Vector(new_n[0][0], new_n[1][0], new_n[2][0]).normalise() * lightDir
					intensity = Vector(screenX, screenY, screenZ).normalize() * lightDir
				elif proj == 'orth':
					screenX, screenY = getOrthographicProjection(new_point[0], new_point[1], new_point[2])
					#intesity = Vector(new_n[0][0], new_n[1][0], new_n[2][0]).normalise() * lightDir
					intensity = Vector(screenX, screenY, p.z).normalize() * lightDir

				
				if intensity < 0:
					cull = True
					break


				if proj == 'pers':
					transformedPoints.append(Point(screenX, screenY, screenZ, Color(intensity*255, intensity*255, intensity*255, 255)))
				elif proj == 'orth':
					transformedPoints.append(Point(screenX, screenY, p.z, Color(intensity*255, intensity*255, intensity*255, 255)))

			if not cull:
				Triangle(transformedPoints[0], transformedPoints[1], transformedPoints[2]).draw(image, zBuffer)

		image.saveAsPNG('image.png')
		im = plt.imread('image.png')
		plt.imshow(im)
		plt.pause(0.000001)


# Calculate face normals
faceNormals = {}
for face in model.faces:
	p0, p1, p2 = [model.vertices[i] for i in face]
	faceNormal = (p2-p0).cross(p1-p0).normalize()

	for i in face:
		if not i in faceNormals:
			faceNormals[i] = []

		faceNormals[i].append(faceNormal)

# Calculate vertex normals
vertexNormals = []
for vertIndex in range(len(model.vertices)):
	vertNorm = getVertexNormal(vertIndex, faceNormals)
	vertexNormals.append(vertNorm)


dataset = pd.read_csv(sys.argv[1])
dataset = updateData(dataset)

projection = sys.argv[2]

mode = sys.argv[3]

output(dataset, projection, mode)