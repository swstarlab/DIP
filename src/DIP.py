#! /usr/bin/env python
# -*- encoding: UTF-8 -*-


'''
DIP : Deep Integrated Perception Framework
ROBOCUP@HOME perception modules for team AUPAIR
Jinyoung Choi
2017.04.25
'''


import sys
import time
import cv2
import numpy as np
import os
from threading import Thread

#ROS modules
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion , Twist, Pose, PoseStamped, Vector3
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image,CameraInfo
from std_msgs.msg import Int32,String,ColorRGBA
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from tf import TransformListener, Transformer, transformations
from dip_jychoi.msg import objs, objs_array, string_array
import DIP_load_config

class DIP():

	def __init__(self): #requires ip and port
		self.params = DIP_load_config.load_config()
		rospy.init_node("DIP")
		self.cvbridge = CvBridge()		
			
		self.captioning_keywords = []
		self.reid_targets = []
		
		self.objs_history = []
		self.objs_history_idx = []

		self.objs_history_max = 10

		self.reid_history = []
		self.reid_history_idx = []

		self.reid_history_max = 3

		self.pose_history = []
		self.pose_history_idx = []

		self.pose_history_max = 3

		self.cap_history = []
		self.cap_history_idx = []

		self.cap_history_max = 3

		self.objects = objs_array()
		self.objects_w_pose = objs_array()
		self.people_identified = objs_array()
		self.reid_targets = objs_array()

		self.sub_obj = rospy.Subscriber(self.params['obj_topic'], objs_array, self.callback_objs, queue_size=1)
		self.sub_obj_w_pose = rospy.Subscriber(self.params['pose_topic'], objs_array, self.callback_pose, queue_size=1)
		self.sub_obj_w_identify = rospy.Subscriber(self.params['reid_topic'], objs_array, self.callback_identify, queue_size=1)
		self.sub_obj_w_cap = rospy.Subscriber(self.params['captioning_topic'], objs_array, self.callback_objs_w_caps, queue_size=1)

		self.reid_target_pub = rospy.Publisher(self.params['reid_target_topic'] ,objs_array,queue_size=1)
		self.captioning_keywords_pub = rospy.Publisher(self.params['captioning_keywords_topic'] ,string_array,queue_size=1)
		self.perception_pub = rospy.Publisher(self.params['perception_topic'] ,objs_array,queue_size=1)
		
		self.perception = objs_array()
		
		#for captioning
		self.pub_cap_req = rospy.Publisher(self.params['captioning_request_topic'],objs_array,queue_size=1)
		self.sub_cap_res = rospy.Subscriber(self.params['captioning_response_topic'], objs_array, self.callback_captioning_response)
		self.captioning_req_time = time.time()
		self.captioning_flag = False
		self.captioning_result = []
		
		if self.params['show_integrated_perception'] :
			cv2.startWindowThread()
			cv2.namedWindow('DIP_jychoi')
		
		while not rospy.is_shutdown():
			self.process_perception()

	def process_perception(self):
		joint_colors = {
			0 : (0,0,255),
			1 : (0,255,0),
			2 : (255,0,0),
			3 : (0,255,255),
			4 : (255,0,255),
			5 : (255,255,0),
			6 : (255,255,255),
			7 : (0,0,100),
			8 : (0,100,0),
			9 : (100,0,0),
			10 : (0,100,100),
			11 : (100,0,100),
			12 : (100,100,0),
			13 : (100,100,100),
			14 : (0,0,50),
			15 : (0,50,0),
			16 : (50,0,0),
			17 : (50,0,50),
		}

		tic = time.time()
		per = self.get_perception(reid=True,pose=True,captioning=True)
		if per.msg_idx == 0 : return None
			
		self.perception_pub.publish(per)
		
		if self.params['show_integrated_perception'] :
			rgb = self.rosimg_to_numpyimg( per.scene_rgb )
			cloud = self.rosimg_to_numpyimg( per.scene_cloud , 'passthrough')
			cloud = cloud/4.96
			mask = (cloud > 0).astype('int')
			cloud = 1.0 - cloud
			cloud *= mask
			cloud *= 255
			cloud = cloud[:,:,0].reshape((240,320,1)).astype('uint8')
			rgbd = (0.7*rgb + 0.3*cloud).astype('uint8')

			for o in per.objects :
				x = o.x - o.h
				y = o.y - o.w
				h = 2*o.h ; w = 2*o.w
				cv2.rectangle(rgbd, (y,x) , (y+w,x+h) , (0,255,0) , 2 ) #bounding box
				tags = sorted(o.tags)
				tags_string = ''
				tags_string += o.class_string
				tags_string += '('
				for t in tags : 
					if t != o.class_string : tags_string += (t + ',')
				tags_string += ')'

				cv2.rectangle(rgbd, (y-1,x-1) , (y+w+1,x+10) , (0,0,0) , -1 )
				cv2.putText(rgbd,tags_string,(y,x+7),cv2.FONT_HERSHEY_SIMPLEX,0.3,color=(255,255,255),thickness=1)

				if len(o.joints) > 0 :
					for i in range(0,18) :
						if o.joints[2*i] > 0 and o.joints[2*i+1] > 0 :
							cv2.circle(rgbd, (y+int(o.joints[2*i+1]),x+int(o.joints[2*i])) , 2, joint_colors[i] , -1)

			cv2.imshow('DIP_jychoi',cv2.resize(rgbd ,(640,480)) )
			


	def get_perception(self,fil = None,reid=True,pose=True,captioning=True):
		 
		objs = self.match_perception(reid,pose,captioning)
		td =  time.time() - objs.header.stamp.secs
		#print td

		if fil is None : return objs
		else :
			result = objs_array()
			result.header = objs.header
			result.msg_idx = objs.msg_idx
			result.scene_rgb = objs.scene_rgb
			result.scene_cloud = objs.scene_cloud

			for item in objs.objects :
				if item.class_string in fil :
					result.objects.append(item)

			return result

	def match_perception(self,reid=True,pose=True,captioning=True):

		tic = time.time()
		required = 0
		required += int(reid) + int(pose) + int(captioning)

		ooo = self.objs_history[::-1]
		if len(ooo) == 0 : return objs_array()

		rrr = self.reid_history[::-1]
		ppp = self.pose_history[::-1]
		ccc = self.cap_history[::-1]

		oooi = self.objs_history_idx[::-1]
		rrri = self.reid_history_idx[::-1]
		pppi = self.pose_history_idx[::-1]
		ccci = self.cap_history_idx[::-1]

		contingency = None

		for k in range( len( oooi ) ) :
			got = 0
			r,p,c = None,None,None
			if (oooi[k] in rrri and reid) :
				r = rrr[  rrri.index(oooi[k])  ] ; got+=1
			if (oooi[k] in pppi and pose) :
				p = ppp[ pppi.index(oooi[k])  ] ; got+=1
			if (oooi[k] in ccci and captioning) :
				c = ccc[ ccci.index(oooi[k])  ] ; got+=1

			for o in ooo[k].objects :
				if o.class_string != 'person' : continue
				if p is not None :
					if contingency is None : contingency = p
					for pp in p.objects :
						if pp.object_id == o.object_id :
							o.joints = pp.joints
							o.tags += pp.tags
							o.pose_wrt_robot = pp.pose_wrt_robot
							o.pose_wrt_odom = pp.pose_wrt_odom
							o.pose_wrt_map = pp.pose_wrt_map

				if r is not None :
					for rr in r.objects :
						if rr.object_id == o.object_id :
							o.person_id = rr.person_id
							o.person_name = rr.person_name
							o.reid_score = rr.reid_score
							o.tags += rr.tags

				if c is not None :
					for cc in c.objects :
						if cc.object_id == o.object_id :
							o.captions = cc.captions
							o.tags += cc.tags

				o.tags = list(set(o.tags))

			if got == required : return ooo[k]
		if contingency is not None : return contingency
		return ooo[ 0  ]

	def callback_objs(self,msg):
		self.objects = msg
		self.objs_history.append(self.objects)
		self.objs_history_idx.append(self.objects.msg_idx)
		if len(self.objs_history) > self.objs_history_max :
			del self.objs_history[  0  ] ; 	del self.objs_history_idx[  0  ]

	def callback_pose(self,msg):
		self.objects_w_pose = msg
		self.pose_history.append(msg)
		self.pose_history_idx.append(msg.msg_idx)
		if len(self.pose_history) > self.pose_history_max :
			del self.pose_history[  0  ] ; 	del self.pose_history_idx[  0  ]

	def callback_identify(self,msg):
		self.people_identified = msg
		self.reid_history.append(msg)
		self.reid_history_idx.append(msg.msg_idx)
		if len(self.reid_history) > self.reid_history_max :
			del self.reid_history[  0  ] ; 	del self.reid_history_idx[  0  ]

	def callback_objs_w_caps(self,msg):
		self.people_captioned = msg
		self.cap_history.append(msg)
		self.cap_history_idx.append(msg.msg_idx)
		if len(self.cap_history) > self.cap_history_max :
			del self.cap_history[  0  ] ; 	del self.cap_history_idx[  0  ]


	def get_captions(self):		

		msg = objs_array()
		msg.scene_rgb = self.numpyimg_to_rosimg( self.get_rgb() )

		self.captioning_req_time = time.time()
		result = ['no description']
		self.captioning_flag = False
		self.pub_cap_req.publish(msg)
		while time.time()-self.captioning_req_time < 3 :
			#print time.time()-self.captioning_req_time
			if self.captioning_flag : result = self.captioning_result ; break
		self.captioning_flag = False
		return result

	def callback_captioning_response(self,msg):
		self.captioning_result = msg.tags
		self.captioning_flag = True
		
	def rosimg_to_numpyimg(self,img_msg,encoding='bgr8'):
		return self.cvbridge.imgmsg_to_cv2(img_msg,encoding)

	def numpyimg_to_rosimg(self,npimg,encoding='bgr8'):
		return self.cvbridge.cv2_to_imgmsg(npimg,encoding)


def main():

	dip = DIP()
	rospy.spin()


if __name__ == "__main__":
	main()
