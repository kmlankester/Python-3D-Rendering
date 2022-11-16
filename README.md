An implementation of a 3D rendering engine with Python using IMU data. This is an extension of https://github.com/ecann/RenderPy

This engine uses a dead-reckoning filter to calculate the current orientation of the object.
Two projection types are available, orthographic projection and perspective projection.
The measurements used can be chosen from the following:
* Gyroscope data
* Gyroscope and accelerometer data
* Gryoscope, acceleromete, and magnetometer data

render.py require three arguments:
* IMU Dataset (string)
* Projection Type (string) - 'pers' or 'orth'
* Mode (string) - 'gyro', 'acc', or 'mag'

To execute render.py run *python render.py dataset projection mode*
