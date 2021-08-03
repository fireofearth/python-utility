import json
import itertools
import functools
import numpy as np
import networkx as nx
import os
import math
import carla

import utility as util

class CARLAUtilException(Exception):
    pass

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

def location_to_ndarray(l, flip_x=False, flip_y=False):
    """Converts carla.Location to ndarray [x, y, z]"""
    x_mult, y_mult = util.decision_to_value([flip_x, flip_y])
    return np.array([x_mult*l.x, y_mult*l.y, l.z])

def rotation_to_ndarray(r, flip_x=False, flip_y=False):
    """Converts carla.Rotation to ndarray [pitch, yaw, roll]"""
    if flip_x and flip_y:
        raise NotImplementedError()
    if flip_x:
        raise NotImplementedError()
    if flip_y:
        pitch, yaw, roll = np.deg2rad([r.pitch, r.yaw, r.roll])
        yaw = util.reflect_radians_about_x_axis(yaw)
        pitch = -pitch
        roll = -roll
        return np.array([pitch, yaw, roll])
    else:
        return np.deg2rad([r.pitch, r.yaw, r.roll])

def locations_to_ndarray(ls, flip_x=False, flip_y=False):
    """Converts list of carla.Location to ndarray of size (len(ls), 3)."""
    return util.map_to_ndarray(
            lambda l: location_to_ndarray(l, flip_x=flip_x, flip_y=flip_y), ls)

def ndarray_to_location(v, flip_x=False, flip_y=False):
    """ndarray of form [x, y, z] to carla.Location."""
    x_mult, y_mult = util.decision_to_value([flip_x, flip_y])
    return carla.Location(x=x_mult*v[0], y=y_mult*v[1], z=v[2])

def actor_to_location_ndarray(a, flip_x=False, flip_y=False):
    """Converts carla.Actor's location to ndarray [x, y, z]"""
    return location_to_ndarray(a.get_location(), flip_x=flip_x, flip_y=flip_y)

def to_location_ndarray(a, flip_x=False, flip_y=False):
    """Converts location of object of relevant carla class to ndarray [x, y, z]"""
    if isinstance(a, carla.Actor):
        return location_to_ndarray(a.get_location(), flip_x=flip_x, flip_y=flip_y)
    elif isinstance(a, carla.Junction):
        return location_to_ndarray(a.bounding_box.location, flip_x=flip_x, flip_y=flip_y)
    elif isinstance(a, carla.Waypoint):
        return location_to_ndarray(a.transform.location, flip_x=flip_x, flip_y=flip_y)
    elif isinstance(a, carla.Transform):
        return location_to_ndarray(a.location, flip_x=flip_x, flip_y=flip_y)
    elif isinstance(a, carla.Location):
        return location_to_ndarray(a, flip_x=flip_x, flip_y=flip_y)
    else:
        raise CARLAUtilException("Not relevant carla class.")

def actor_to_velocity_ndarray(a, flip_x=False, flip_y=False):
    """Converts carla.Actor's component-wise velocity to ndarray
    [vel_x, vel_y, vel_z]"""
    x_mult, y_mult = util.decision_to_value([flip_x, flip_y])
    v = a.get_velocity()
    return np.array([x_mult*v.x, y_mult*v.y, v.z])

def actor_to_bbox_ndarray(a):
    """Converts carla.Actor's bounding box dimensions to ndarray
    [bbox_x, bbox_y, box_z]. bbox_x is the length on the longitudinal axis,
    bbox_y is the length on the lateral axis."""
    bb = a.bounding_box.extent
    return np.array([2*bb.x, 2*bb.y, 2*bb.z])

def actor_to_rotation_ndarray(a, flip_x=False, flip_y=False):
    """Converts carla.Actor's component-wise velocity to ndarray
    [pitch, yaw, roll]"""
    t = a.get_transform()
    r = t.rotation
    return rotation_to_ndarray(r, flip_x=flip_x, flip_y=flip_y)

def to_rotation_ndarray(a, flip_x=False, flip_y=False):
    """Converts velocity of object of relevant carla class to ndarray
    [pitch, yaw, roll]"""
    if isinstance(a, carla.Actor):
        return rotation_to_ndarray(a.get_transform().rotation, flip_x=flip_x, flip_y=flip_y)
    elif isinstance(a, carla.Waypoint):
        return rotation_to_ndarray(a.transform.rotation, flip_x=flip_x, flip_y=flip_y)
    elif isinstance(a, carla.Transform):
        return rotation_to_ndarray(a.rotation, flip_x=flip_x, flip_y=flip_y)
    else:
        raise CARLAUtilException("Not relevant carla class.")

def actor_to_Lxyz_Vxyz_Axyz_Rpyr_ndarray(a, flip_x=False, flip_y=False):
    """Converts carla.Vehicle
    to ndarray [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z,
    length, width, height, pitch, yaw, roll] where pitch, yaw, roll are in
    radians."""
    x_mult, y_mult = util.decision_to_value([flip_x, flip_y])
    bb = a.bounding_box.extent
    t = a.get_transform()
    v = a.get_velocity()
    a = a.get_acceleration()
    l = t.location
    r = t.rotation
    l = [x_mult*l.x, y_mult*l.y, l.z]
    v = [x_mult*v.x, y_mult*v.y, v.z]
    a = [x_mult*a.x, y_mult*a.y, a.z]
    bb = [2*bb.x, 2*bb.y, 2*bb.z]
    r = rotation_to_ndarray(r, flip_x=flip_x, flip_y=flip_y)
    return np.concatenate((l, v, a, bb, r))

def actors_to_location_ndarray(alist, flip_x=False, flip_y=False):
    """Converts iterable of carla.Actor to a ndarray of size (len(alist), 3)"""
    return util.map_to_ndarray(
            lambda a: actor_to_location_ndarray(a, flip_x=flip_x, flip_y=flip_y), alist)

def actors_to_Lxyz_Vxyz_Axyz_Rpyr_ndarray(alist, flip_x=False, flip_y=False):
    """Converts iterable of carla.Actor transformation
    to an ndarray of size (len(alist), 15) where each row is
    [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z,
    length, width, height, pitch, yaw, roll]
    where pitch, yaw, roll are in radians."""
    return util.map_to_ndarray(
            lambda a: actor_to_Lxyz_Vxyz_Axyz_Rpyr_ndarray(a, flip_x=flip_x, flip_y=flip_y), alist)

def transform_to_location_ndarray(t, flip_x=False, flip_y=False):
    """Converts carla.Transform to location ndarray [x, y, z]"""
    return location_to_ndarray(t.location, flip_x=flip_x, flip_y=flip_y)

def transform_to_yaw(t):
    """Converts carla.Transform to rotation yaw mod 360"""
    return t.rotation.yaw % 360.

def transforms_to_location_ndarray(ts, flip_x=False, flip_y=False):
    """Converts an iterable of carla.Transform to a ndarray of size (len(iterable), 3)"""
    return util.map_to_ndarray(
            lambda t: transform_to_location_ndarray(t, flip_x=flip_x, flip_y=flip_y), ts)

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

# Scratch work
##############

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
