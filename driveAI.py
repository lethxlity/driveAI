import carla

class Simulation:
    settings = {'map': 'Town04',
                'vehicle_name': 'vehicle.dodge.charger_2020',
                'sensors': {'sensor.camera.rgb': {'image_size_x': '1600',
                                                  'image_size_y': '590',
                                                  'fov': '100'},
                            'sensor.camera.depth': {'image_size_x': '1600',
                                                  'image_size_y': '590',
                                                  'fov': '100'}
                            }
                }

    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.world = self.client.get_world()
        self.spectator = None
        self.vehicle = None
        self.sensors = {}

    def initialize(self, *args, **kwargs):
        print(self.world.get_settings())
        # self.world = self.client.load_world(Simulation.settings['map'])
        self.spectator = self.world.get_spectator()

        vehicle_bps = self.world.get_blueprint_library().filter('*vehicle*')
        vehicle_bp = vehicle_bps.find(Simulation.settings['vehicle_name'])

        spawn_points = self.world.get_map().get_spawn_points()
        self.vehicle = self.world.try_spawn_actor(vehicle_bp,
                                                  spawn_points[10])

        for sensor_name in Simulation.settings['sensors'].keys():
            sensor_bp = self.world.get_blueprint_library().find(sensor_name)
            sensor_bp.set_attribute('image_size_x', Simulation.settings['sensors'][sensor_name]['image_size_x'])
            sensor_bp.set_attribute('image_size_y', Simulation.settings['sensors'][sensor_name]['image_size_y'])
            sensor_bp.set_attribute('fov',          Simulation.settings['sensors'][sensor_name]['fov'])
            self.sensors[sensor_name] = self.world.spawn_actor(sensor_bp,
                                                               carla.Transform(carla.Location(x=0.8, z=1.3)),
                                                               attach_to=self.vehicle)


