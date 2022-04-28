# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
本文件允许模块包以python -m trustai方式直接执行。

Authors: zhangshuai28(zhangshuai28@baidu.com)
Date:    2022/03/14 14:53:37
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


import sys
from trustai.cmdline import main
sys.exit(main())
