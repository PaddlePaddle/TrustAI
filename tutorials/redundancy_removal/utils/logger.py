#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import time


class Logger():

    def __init__(self, args):
        self.args = args
        self.tic_train = time.time()
        self.performance_list = []
        self.mode_dict = {
            "ERROR": "31",
            "INFO": "32",
            "WARNING": "33",
        }

    def __get_time_str(self):
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "," + str(
            int(round(time.time() * 1000)) % 1000).zfill(3)

    def __log(self, log_str, mode="INFO"):
        print("\033[" + self.mode_dict[mode] + "m[" + self.__get_time_str() + "] [    " + mode + "]\033[0m - " +
              log_str)

    def info(self, info_str: str):
        self.__log(info_str, mode="INFO")

    def error(self, info_str: str):
        self.__log(info_str, mode="ERROR")

    def warn(self, info_str: str):
        self.__log(info_str, mode="WARNING")

    def logging_step(self, global_step, epoch, step, loss):
        if global_step % self.args.logging_steps == 0:
            self.info("Global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s" %
                      (global_step, epoch + 1, step + 1, loss, self.args.logging_steps /
                       (time.time() - self.tic_train)))
            self.tic_train = time.time()

    def add_performance(self, performance):
        self.performance_list.append(performance)

    def save_performance(self):
        output_dir = os.path.join(self.args.output_dir, "logging.json")
        with open(output_dir, "w", encoding="utf8") as f:
            json.dump(self.performance_list, f)

    def logging_result(self, x):
        print(x)
