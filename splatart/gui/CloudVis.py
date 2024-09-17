# This app is derived heavily from the Open3D example at
# https://www.open3d.org/docs/latest/python_example/visualization/index.html

# The Open3D example is licensed under MIT, as is this file.


import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np

class CloudVisApp:

    def __init__(self):
        self._id = 0
        self.window = gui.Application.instance.create_window(
            "CloudVisualization", 1024, 768)
        em = self.window.theme.font_size # used for setting margins on widgets

        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([1, 1, 1, 1])
        self.scene.scene.scene.set_sun_light(
            [-1, -1, -1],  # direction
            [1, 1, 1],  # color
            100000)  # intensity
        self.scene.scene.scene.enable_sun_light(True)

        bbox = o3d.geometry.AxisAlignedBoundingBox([-10, -10, -10],
                                                   [10, 10, 10])
        self.scene.setup_camera(10, bbox, [0, 0, 0])

        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(1.0)
        self.material = o3d.visualization.rendering.MaterialRecord()
        self.material.shader = "defaultLit"
        self.scene.scene.add_geometry("coord_frame", self.coord_frame, self.material)

        self.collapse = gui.CollapsableVert("Widgets", 0.33 * em, gui.Margins(em, 0, 0, 0))
        self._label = gui.Label("Lorem ipsum dolor")
        self._label.text_color = gui.Color(1.0, 0.5, 0.0)
        self.collapse.add_child(self._label)

        
        self.window.add_child(self.scene)
        self.window.add_child(self.collapse)
        self.window.set_on_layout(self._on_layout)

        self.sliders = {}

    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        r = self.window.content_rect
        self.scene.frame = r

        width = r.width

        height = min(
            r.height,
            self.collapse.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        
        self.collapse.frame = gui.Rect(0, 0, width,
                                              height)
       
    def add_cloud(self, cloud, name:str = None, material:o3d.visualization.rendering.MaterialRecord = None):
        if(name is None):
            name = f"cloud_{self._id}"
            self._id += 1
        print(f"adding cloud: {name}")
        self.scene.scene.add_geometry(name, cloud, material if material is not None else self.material)
        # self.window.update_renderer()

    def update_cloud(self, cloud, name:str):
        print(f"updating cloud: {name}")
        # get the existing cloud
        self.scene.scene.remove_geometry(name)
        self.scene.scene.add_geometry(name, cloud, self.material)
        self.scene.force_redraw()

    def add_slider(self, name:str, min_val:float, max_val:float, val:float, callback_fn):
        slider = gui.Slider(gui.Slider.DOUBLE)
        slider.set_limits(min_val, max_val)
        slider.double_value = val
        slider.set_on_value_changed(callback_fn)
        self.sliders[name] = slider
        self.collapse.add_child(slider)

def main():
    gui.Application.instance.initialize()
    app = CloudVisApp()
    # app.add_cloud(o3d.geometry.TriangleMesh.create_sphere(1.0), "sphere")
    test_cloud = o3d.geometry.PointCloud()
    test_cloud.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))
    test_cloud.paint_uniform_color([1, 0, 1])
    app.add_cloud(test_cloud, "test_cloud")
    gui.Application.instance.run()

if __name__ == "__main__":
    main()