from honcho.manager import Manager

manager = Manager()
manager.add_process('backend', 'python server.py --config conf/local.conf')
manager.add_process('frontend', 'npm start')

manager.loop()
