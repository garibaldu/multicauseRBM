import logging



class Progress(object):

    def __init__(self, name, total_units):
        self.name = name
        self.total_units = float(total_units)
        self.completed_units = 0.0
        self.update_count = 0
        self.progess_logger = logging.getLogger(name)
        self.progess_logger.info("Created Progress logger for task - {}".format(name))

    def set_completed_units(self, completed_units):
        self.completed_units = completed_units
        self.update_count += 1

        if self.percent_complete() % self.percent_update == 0:
            self.report()


    def percent_complete(self):
        # get the percentage of the children
        return (self.completed_units / self.total_units) * 100

    def set_percentage_update_frequency(self, percent):
        self.percent_update = percent

    def report(self):
        self.progess_logger.info("{}% complete".format(self.percent_complete()))

    def finished(self):
        self.progess_logger.info("100% complete")
