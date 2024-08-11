# Adobe-GFG-Solution

This solution is submitted by ByteSquadron
Team members are:
Harpreet Singh,
Gurpreet Singh,
Rashmeet Singh.

#The solution description is as follows:

Geometric Shape Detection and Regularization:

The code is designed to analyze paths composed of point coordinates and identify basic geometric shapes such as circles, ellipses, straight lines, polygons, and star shapes. For each shape, the code applies specific algorithms:

Circle Detection: Uses curve fitting to find the best-fit circle, then regularizes the curve to form a perfect circle.

Ellipse Detection: Uses singular value decomposition (SVD) to fit an ellipse to the data points.

Straight Line Detection: Checks if the points form a straight line by calculating perpendicular distances from the points to the line.

Polygon and Star Shape Detection: Uses ConvexHull to detect if points form a regular polygon or a star shape by analyzing angles and distances.

Curve Completion:
The code handles incomplete curves by using cubic interpolation. If a curve is not closed, it extrapolates additional points to complete the shape and closes it by connecting the endpoints.

Symmetry Detection:
The code detects symmetries by identifying point pairs that are symmetric with respect to the center of the shape.

Visualization:
Processed paths are visualized using Matplotlib, with different colors assigned to each path for easy differentiation.

CSV Integration:
Paths are read from a CSV file where each path's points are organized by indices. The code can handle multiple paths and regularize or complete them as needed.

This comprehensive solution is suitable for applications that require analysis and regularization of geometric shapes in datasets.
