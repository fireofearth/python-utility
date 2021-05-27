import json
import itertools
import functools
import numpy as np
import networkx as nx
import os
import carla

import utility as util

def make_client(host='127.0.0.1', port=2000):
    """Create a client. Useful for debugging in the Python interpreter."""
    return carla.Client(host, port)

def debug_point(client, l, t=1.0,
        c=carla.Color(r=255, g=0, b=0, a=100)):
    """Draw a point in the simulator.

    Parameters
    ----------
    client : carla.Client or carla.World
        Client.
    l : carla.Transform or carla.Location
        Position in map to place the point.
    t : float, optional
        Life time of point.
    c : carla.Color, optional
        Color of the point.
    """
    if isinstance(l, carla.Transform):
        l = l.location
    if isinstance(client, carla.Client):
        world = client.get_world()
    else:
        world = client
    world.debug.draw_string(l, 'o', color=c, life_time=t)

def debug_square(client, l, r,
        rotation=carla.Rotation(), t=1.0,
        c=carla.Color(r=255, g=0, b=0, a=100)):
    """Draw a square centered on a point.

    Parameters
    ----------
    client : carla.Client or carla.World
        Client.
    l : carla.Transform or carla.Location
        Position in map to place the point.
    r : float
        Radius of the square from the center
    t : float, optional
        Life time of point.
    c : carla.Color, optional
        Color of the point.
    """
    if isinstance(l, carla.Transform):
        l = l.location
    if isinstance(client, carla.Client):
        world = client.get_world()
    else:
        world = client
    box = carla.BoundingBox(l, carla.Vector3D(r, r, r))
    world.debug.draw_box(box, rotation, thickness=0.5,
            color=c, life_time=t)

def location_to_ndarray(l):
    """Converts carla.Location to ndarray [x, y, z]"""
    return np.array([l.x, l.y, l.z])

def ndarray_to_location(v):
    """ndarray of form [x, y, z] to carla.Location."""
    return carla.Location(x=v[0], y=v[1], z=v[2])

def actor_to_location_ndarray(a):
    """Converts carla.Actor's location to ndarray [x, y, z]"""
    return location_to_ndarray(a.get_location())

def vehicle_to_xyz_lwh_pyr_ndarray(a):
    """Converts carla.Vehicle's transformation
    to ndarray [x, y, z, length, width, height, pitch, yaw, roll]"""
    t = a.get_transform()
    l = t.location
    r = t.rotation
    r = np.deg2rad([r.pitch, r.yaw, r.roll])
    bb = a.bounding_box.extent
    return np.array([l.x, l.y, l.z, 2*bb.x, 2*bb.y, 2*bb.z, *r])

def actors_to_location_ndarray(alist):
    """Converts iterable of carla.Actor to a ndarray of size (len(alist), 3)"""
    return np.array(util.map_to_list(actor_to_location_ndarray, alist))

def vehicles_to_xyz_lwh_pyr_ndarray(alist):
    """Converts iterable of carla.Actor transformation
    to an ndarray of size (len(alist), 9) where each row is
    [x, y, z, length, width, height, pitch, yaw, roll]"""
    return np.array(util.map_to_list(vehicle_to_xyz_lwh_pyr_ndarray, alist))

def transform_to_location_ndarray(t):
    """Converts carla.Transform to location ndarray [x, y, z]"""
    return location_to_ndarray(t.location)

def tranform_to_xyz_pyr_ndarray(t):
    """Converts carla.Transform to
    ndarray [x, y, z, pitch, yaw, roll]"""
    l = t.location
    r = t.rotation
    r = np.deg2rad([r.pitch, r.yaw, r.roll])
    return np.array([l.x, l.y, l.z, *r])

def transform_to_yaw(t):
    """Converts carla.Transform to rotation yaw mod 360"""
    return t.rotation.yaw % 360.

def transforms_to_location_ndarray(ts):
    """Converts an iterable of carla.Transform to a ndarray of size (len(iterable), 3)"""
    return np.array(util.map_to_list(transform_to_location_ndarray, ts))


def transform_points(transform, points):
    """Given a 4x4 transformation matrix, transform an array of 3D points.
    Expected point foramt: [[X0,Y0,Z0],..[Xn,Yn,Zn]]."""
    # Needed format: [[X0,..Xn],[Z0,..Zn],[Z0,..Zn]]. So let's transpose
    # the point matrix.
    points = points.transpose()
    # Add 0s row: [[X0..,Xn],[Y0..,Yn],[Z0..,Zn],[0,..0]]
    points = np.append(points, np.ones((1, points.shape[1])), axis=0)
    # Point transformation
    # points = transform * points
    points = np.dot(transform, points)
    # Return all but last row
    return points[0:3].transpose()

def transform_to_origin(transform, origin):
    """Create an adjusted transformation relative to origin.
    Creates a new transformation (doesn't mutate the parameters).
    
    Parameters
    ----------
    transform : carla.Transform
        The transform we want to adjust
    origin : carla.Transform or np.array
        The origin we want to adjust the transform to

    Returns
    -------
    carla.Transform
        New transform with the origin as reference.
    """
    location = transform.location
    rotation = transform.rotation
    return carla.Transform(
            carla.Location(
                x=location.x - origin.location.x,
                y=location.y - origin.location.y,
                z=location.z - origin.location.z),
            carla.Rotation(
                pitch=rotation.pitch,
                yaw=rotation.yaw - origin.rotation.yaw,
                roll=rotation.roll))

def get_junctions_from_topology_graph(topology):
    """Gets unique junctions from topology

    Parameters
    ----------
    topology : nx.Graph or list

    Returns
    -------
    list of carla.Junction
    """
    if isinstance(topology, list):
        G = nx.Graph()
        G.add_edges_from(topology)
        topology = G
    junctions = map(lambda v: v.get_junction(),
        filter(lambda v: v.is_junction, topology.nodes))
    return list({j.id: j for j in junctions}.values())

"""Creating internal library for similarity transformations
since CARLA transformation matrices are unreliable.

Based on
https://cs184.eecs.berkeley.edu/uploads/lectures/05_transforms-2/05_transforms-2_slides.pdf
"""

def create_translation_mtx(x, y, z):
    return np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]])

def create_x_rotation_mtx(a):
    """a is roll"""
    return np.array([
            [1, 0,          0,         0],
            [0, np.cos(a), -np.sin(a), 0],
            [0, np.sin(a),  np.cos(a), 0],
            [0, 0,          0,         1]])

def create_y_rotation_mtx(b):
    """b is pitch"""
    return np.array([
            [ np.cos(b), 0, np.sin(b), 0],
            [ 0,         1, 0,         0],
            [-np.sin(b), 0, np.cos(b), 0],
            [ 0,         0, 0,         1]])

def create_z_rotation_mtx(c):
    """c is yaw"""
    return np.array([
            [np.cos(c), -np.sin(c), 0, 0],
            [np.sin(c),  np.cos(c), 0, 0],
            [0,          0,         1, 0],
            [0,          0,         0, 1]])

# Scratch work
##############

def transform_to_translation_mtx(transform):
    l = transform.location
    return create_translation_mtx(l.x, l.y, l.z)

class SimilarityTransform(object):

    @staticmethod
    def to_radians(d):
        return math.radians(d)

    @classmethod
    def from_transform(cls, transform):
        location = transform.location
        rotation = transform.rotation
        return cls(location.x, location.y, location.z,
                cls.to_radians(rotation.yaw),
                cls.to_radians(rotation.pitch),
                cls.to_radians(rotation.roll))

    def __init__(self, x, y, z, yaw, pitch, roll):
        self.x = x
        self.y = y
        self.z = z
        self.translation = create_translation_mtx(x, y, z)
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
