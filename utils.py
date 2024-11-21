import numpy as np
import logging 
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from collections import deque

def read_pose(data_path:str)->np.ndarray:
    pose = np.load(data_path)
    pose = pose['poses'][:,:72]
    pose[:,66:72] = 0.0 # reset hand
    #logging.debug(f' pose shape : {pose.shape}')
    return pose

def read_beta(data_path:str)->np.ndarray:
    betas = np.load(data_path)
    betas = betas['betas'][:,:10]
    return betas

def load_obj(file_obj:str):
    vertices = []
    faces = []
    with open(file_obj, 'r') as file:
        for line in  file:
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'v':
                vertex = np.array([float(values[1]), float(values[2]), float(values[3])])
                vertices.append(vertex)
            if values[0] == 'f':
                face = [int(value.split('/')[0]) - 1 for value in values[1:]]
                faces.append(face)
    vertices = np.array(vertices,dtype=np.float64)
    faces = np.array(faces,dtype=np.int32)
    return vertices, faces

class Mesh():
    def __init__(self, vertices, faces, compute_edges=True, compute_vertex_normals=True)-> None:
        self.vertices = vertices 
        self.faces = faces 
        self.edges=[]
        self.vertex_normals=[]
        self.adjacency_matrix = []
        if compute_edges:
            self.compute_edges()
        if compute_vertex_normals:
            self.compute_vertex_normals()
    
    def compute_edges(self, directed=False):
        edges=set()
        num_vertices=3 # triangle mesh
        for f in self.faces:
            for i in range(num_vertices):
                # get the two vertices form the edge
                v1 = f[i]
                v2 = f[(i+1)%num_vertices] 
                # sort the vertices to ensure consistency(avoid duplicates)
                edge=tuple(sorted((v1, v2)))
                edges.add(edge)
        self.edges = np.array(list(edges)).T 
        if directed is not True:
            row1 = np.append(self.edges[0], self.edges[1])
            row2 = np.append(self.edges[1], self.edges[0])
            self.edges = np.append(row1, row2).reshape(2,-1)
        return self.edges
    
    '''
        Compute the vertex normal for each vertex in a triangle mesh.

        Parameters:
            vertices (ndarray): A numpy array of shape (N, 3) representing the vertices of the triangle mesh.
            faces (ndarray): A numpy array of shape (M, 3) representing the indices of the vertices of
                             each triangle in the mesh.

        Returns:
            ndarray: A numpy array of shape (N, 3), representing the normal vector of each vertex in the mesh.
    '''
    def compute_vertex_normals(self):
        # Compute the face normal vector for each triangle.
        vertices = self.vertices
        faces = self.faces
        a = vertices[faces[:, 0]]
        b = vertices[faces[:, 1]]
        c = vertices[faces[:, 2]]
        face_normals = np.cross(b - a, c - a)

        # Initialize an array to hold the sum of face normals for each vertex.
        vertex_normals = np.zeros_like(vertices)

        # Add the face normal vectors of each triangle to the corresponding vertices.
        np.add.at(vertex_normals, faces[:, 0], face_normals)
        np.add.at(vertex_normals, faces[:, 1], face_normals)
        np.add.at(vertex_normals, faces[:, 2], face_normals)

        # Normalize each vertex normal vector.
        norms = np.linalg.norm(vertex_normals, axis=1)
        vertex_normals /= np.where(norms == 0, 1, norms)[:, np.newaxis]

        # Make sure each vertex normal vector is anti-clockwise.
        for f in range(faces.shape[0]):
            indices = faces[f]
            face_normal = face_normals[f]
            for i in range(3):
                j, k = indices[i], indices[(i+1)%3]
                angle = np.arccos(np.clip(np.dot(vertex_normals[j], vertex_normals[k]), -1, 1))
                if np.cross(vertex_normals[j], vertex_normals[k]).dot(face_normal) < 0:
                    angle = -angle
                vertex_normals[j] += face_normal * angle / 2
                vertex_normals[k] += face_normal * angle / 2

        # Normalize each vertex normal vector.
        norms = np.linalg.norm(vertex_normals, axis=1)
        vertex_normals /= np.where(norms == 0, 1, norms)[:, np.newaxis]
    
        self.vertex_normals = vertex_normals
        return vertex_normals
    
    def compute_face_normals(self):
        """
        Calculates the normal vector of a triangle mesh given its vertices and faces.

        Parameters:
        vertices (np.ndarray): Array of shape (N, 3) containing the vertices of the triangle mesh.
        faces (np.ndarray): Array of shape (M, 3) containing the indices of the vertices that form the triangles.

        Returns:
        np.ndarray: Normal vector of the triangle mesh, oriented consistently in a counter-clockwise direction. 
        """
        # Get the vertices for each triangle
        vertices = self.vertices
        faces = self.faces
        triangle_vertices = vertices[faces]

        edges = triangle_vertices[:,1:] - triangle_vertices[:,:-1]

        triangle_normals = np.cross(edges[:,0], edges[:,1])
        triangle_normals /= np.linalg.norm(triangle_normals, axis=1)[:, np.newaxis]
    
        return triangle_normals 

    def compute_adjacency_matrix(self, format='csr'):
        if self.edges == []:
            self.compute_edges() 
        dim_adj = len(self.vertices)
        adj_val = np.ones(shape=(len(self.edges[0]),), dtype=np.int)   
        row = self.edges[0]
        col = self.edges[1]
        if format=='csr':
            self.adjacency_matrix = csr_matrix((adj_val, (row, col)), shape=(dim_adj, dim_adj))
        elif format=='csc':
            self.adjacency_matrix = csc_matrix((adj_val, (row, col)), shape=(dim_adj, dim_adj))
        else:
            print('format error, sparse matrix should either csc or csr.')
            self.adjacency_matrix = []
        return self.adjacency_matrix
    # Find neighbours of vertex in within certain radius (step), each edge weight is 1
    def neighbours(self, vertex_index, edge_index, depth):
        
        # Initialize the neighbors list and the BFS queue
        neighbors = set()
        queue = deque([(vertex_index, 0)])  # (vertex, steps)

        while queue:
            current_vertex, steps = queue.popleft()

            if steps > depth:
                break

            # Add the current vertex to neighbors if it's not the original vertex
            if current_vertex != vertex_index:
                neighbors.add(current_vertex)

            # Explore the neighbors of the current vertex
            for neighbor in edge_index[1][edge_index[0] == current_vertex]:
                queue.append((neighbor, steps + 1))

        return neighbors

if __name__=='__main__':
    logging.basicConfig(level=logging.DEBUG)
    data_path = '/home/cxh/Downloads/dataset/PoseShapeSet/instance000/jumping_poses/tshirt_0017.obj'
    v, f = load_obj(data_path)
    logging.debug(f' vertices : {v.shape}')
    logging.debug(f' faces : {f.shape}')
