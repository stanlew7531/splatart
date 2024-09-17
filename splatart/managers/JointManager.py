import torch
import warnings
import xml.etree.ElementTree as ET

from splatart.networks.NarfJoint import RevoluteJoint, PrismaticJoint

class JointManager():
    def __init__(self) -> None:
        self.joints = {}
        self.links = {}

    def joints_from_urdf(self, urdf_fname:str):
        tree = ET.parse(urdf_fname)
        root = tree.getroot()

        joints = root.findall("joint")
        links = root.findall("link")

        # get all links first
        for link in links:
            link_name = link.attrib["name"]
            print(f"Link name: {link_name}")
            if(link_name in self.links):
                raise ValueError(f"Link name {link_name} already exists")
            
            # get the origin with priority for visual/inertial/collision (in that order)
            visual = link.find("visual")
            inertial = link.find("inertial")
            collision = link.find("collision")
            if(visual is not None):
                origin = visual.find("origin")
            elif(inertial is not None):
                origin = inertial.find("origin")
            elif(collision is not None):
                origin = collision.find("origin")
            else:
                raise ValueError(f"No origin found for link: {link_name}")
            
            if(origin is None):
                warnings.warn(f"No origin found for link: {link_name}, proceeding with identity")
                origin_xyz = [0, 0, 0]
                origin_rpy = [0, 0, 0]
            else:
                origin_xyz = origin.attrib["xyz"]
                origin_xyz = [float(x) for x in origin_xyz.split(" ")]
                origin_rpy = origin.attrib["rpy"]
                origin_rpy = [float(x) for x in origin_rpy.split(" ")]

            self.links[link_name] = {"origin_xyz": origin_xyz, "origin_rpy": origin_rpy}

        for joint in joints:
            # check joint type
            joint_type = joint.attrib["type"]
            joint_name = joint.attrib["name"]

            if(joint_name in self.joints):
                raise ValueError(f"Joint name {joint_name} already exists")

            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]

            origin_xyz = [float(x) for x in joint.find("origin").attrib["xyz"].split(" ")]
            origin_rpy = [float(x) for x in joint.find("origin").attrib["rpy"].split(" ")]
                
            if(joint_type == "revolute"):
                print(f"Revolute joint: {joint_name}")
                rev_joint = RevoluteJoint(1) # default to 1 scene for now
                rev_joint.pre_tf = torch.nn.Parameter(torch.Tensor([origin_xyz, origin_rpy]).flatten()) # wrap in parameter here
                print(f"joint origin xyz: {origin_xyz}, rpy: {origin_rpy}")
                # get the child link to determine the post tf
                child_link = self.links[child]
                child_origin_xyz = child_link["origin_xyz"]
                child_origin_rpy = child_link["origin_rpy"]
                rev_joint.post_tf = torch.nn.Parameter(torch.Tensor([child_origin_xyz, child_origin_rpy]).flatten())
                print(f"joint post tf xyz: {child_origin_xyz}, rpy: {child_origin_rpy}")
                self.joints[joint_name] = rev_joint

            if(joint_type == "prismatic"):
                print(f"Prismatic joint: {joint_name}")
                prs_joint = PrismaticJoint(1)
                prs_joint.pre_tf = torch.nn.Parameter(torch.Tensor([origin_xyz, origin_rpy]).flatten())
                prs_joint.joint_axis = torch.nn.Parameter(torch.Tensor([float(x) for x in joint.find("axis").attrib["xyz"].split(" ")]).flatten())
                self.joints[joint_name] = prs_joint
