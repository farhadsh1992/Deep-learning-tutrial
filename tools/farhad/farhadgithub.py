#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 11:32:49 2018

@author: Farhad
"""

import base64
from github import Github
from github import InputGitTreeElement

user = "farhadsh1992"
password = "*********"
g = Github(user,password)
repo = g.get_user().get_repo('git-test')
file_list = [
    '/anaconda3/lib/python3.7/farhad/f_textcleaner.py'
    
]

file_names = [
    'f_textcleaner.py'
]
commit_message = 'python update 2'
master_ref = repo.get_git_ref('heads/master/twitter-analysis')
master_sha = master_ref.object.sha
base_tree = repo.get_git_tree(master_sha)
element_list = list()
for i, entry in enumerate(file_list):
    with open(entry) as input_file:
        data = input_file.read()
    if entry.endswith('.png'):
        data = base64.b64encode(data)
    element = InputGitTreeElement(file_names[i], '100644', 'blob', data)
    element_list.append(element)
tree = repo.create_git_tree(element_list, base_tree)
parent = repo.get_git_commit(master_sha)
commit = repo.create_git_commit(commit_message, tree, [parent])
master_ref.edit(commit.sha)