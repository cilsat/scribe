#!/usr/bin/env python

import graphviz as gv


def dfd_labeling_single(name='dfd_labeling_single', form='png',
                        label='Data flow diagram for labeling a meeting'):
    dot = gv.Digraph(comment=label, format=form)
    # Data nodes
    dot.attr('node', shape='box')
    dot.node('d1', 'Audio recording')
    dot.node('d2', 'WAV recording')
    dot.node('d3', 'Segments data')
    dot.node('d4', 'Speaker-labeled segments')

    # Process nodes
    dot.attr('node', shape='ellipse')
    dot.node('p1', 'Convert')
    dot.node('p2', 'Segment')
    dot.node('p3', 'Label')

    dot.edges([('d1', 'p1'), ('p1', 'd2'), ('d2', 'p2'),
               ('p2', 'd3'), ('d3', 'p3'), ('p3', 'd4')])

    dot.render(name, cleanup=True)


def dfd_labeling_all(name='dfd_labeling_all', form='png',
                     label='Data flow diagram for labeling all meetings'):
    dot = gv.Digraph(comment=label, format=form)

    # Data nodes
    dot.attr('node', shape='box')
    dot.node('d1', 'Audio recordings')
    dot.node('d2', 'Locally-labeled segments')
    dot.node('d3', 'Globally-labeled segments')
    dot.node('d4', 'Speaker hypotheses')

    # Process nodes
    dot.attr('node', shape='ellipse')
    dot.node('p1', 'Label single meeting locally')
    dot.node('p2', 'Label meetings globally')
    dot.node('p3', 'Generate baseline')

    dot.edges([('d1', 'p1'), ('p1', 'd2'), ('d2', 'p2'), ('p2', 'd3'),
               ('d3', 'p3'), ('d1', 'p3'), ('p3', 'd4'), ('d4', 'p2')])
    dot.render(name, cleanup=True)


def dfd_baseline(name='dfd_baseline', form='png',
                 label='Data flow diagram for speaker model training'):
    dot = gv.Digraph(comment=label, format=form)
    # Data nodes
    dot.attr('node', shape='box')
    dot.node('d1', 'Speaker-labeled segments data')
    dot.node('d2', 'Training segments data')
    dot.node('d3', 'Testing segments data')
    dot.node('d4', 'WAV recordings')
    dot.node('d5', 'Training segments audio')
    dot.node('d6', 'Testing segments audio')
    dot.node('d7', 'Universal background model')
    dot.node('d8', 'Speaker model')
    dot.node('d9', 'Speaker hypotheses')
    dot.node('d10', 'DER results')

    # Process nodes
    dot.attr('node', shape='ellipse')
    dot.node('p1', 'Allocate train/test segments')
    dot.node('p2', 'Slice audio segment')
    dot.node('p3', 'GMM training')
    dot.node('p4', 'MAP adaptation')
    dot.node('p5', 'Speaker identification')
    dot.node('p6', 'Calculate DER')

    dot.edges([('d1', 'p1'), ('p1', 'd2'), ('p1', 'd3'), ('d2', 'p2'),
               ('d3', 'p2'), ('d4', 'p2'), ('p2', 'd5'), ('p2', 'd6'),
               ('d5', 'p3'), ('d2', 'p3'), ('p3', 'p4'), ('d7', 'p4'),
               ('p4', 'd8'), ('d8', 'p5'), ('d6', 'p5'), ('d3', 'p5'),
               ('p5', 'd9'), ('d9', 'p6'), ('d2', 'p6'), ('p6', 'd10')])

    dot.render(name, cleanup=True)


def dfd_offline(name='dfd_offline', form='png',
                label='Data flow diagram for offline identification system'):
    dot = gv.Digraph(comment=label, format=form)
    # Data nodes
    dot.attr('node', shape='box')
    dot.node('d1', 'Audio recording')
    dot.node('d2', 'Segments')
    dot.node('d3', 'Speaker model')
    dot.node('d4', 'ASR models')
    dot.node('d5', 'Speaker hypotheses')
    dot.node('d6', 'Speech hypotheses')
    dot.node('d7', 'Transcript')

    # Process nodes
    dot.attr('node', shape='ellipse')
    dot.node('p1', 'Segment')
    dot.node('p2', 'Identify speakers')
    dot.node('p3', 'Identify speech')
    dot.node('p4', 'Generate transcript')

    dot.edges([('d1', 'p1'), ('p1', 'd2'), ('d2', 'p2'), ('d3', 'p2'),
               ('p2', 'd5'), ('d1', 'p3'), ('d4', 'p3'), ('p3', 'd6'),
               ('d5', 'p4'), ('d6', 'p4'), ('p4', 'd7')])
    dot.render(name, cleanup=True)


def dfd_online(name='dfd_online', form='png',
               label='Data flow diagram for online identification system'):
    dot = gv.Digraph(comment=label, format=form)
    # Data nodes
    dot.attr('node', shape='box')
    dot.node('d1', 'Audio stream')
    dot.node('d2', 'Segments')
    dot.node('d3', 'Speaker model')
    dot.node('d4', 'ASR models')
    dot.node('d5', 'Speaker hypotheses')
    dot.node('d6', 'Speech hypotheses')
    dot.node('d7', 'Transcript')

    # Process nodes
    dot.attr('node', shape='ellipse')
    dot.node('p1', 'Segment')
    dot.node('p2', 'Identify speakers')
    dot.node('p3', 'Identify speech')
    dot.node('p4', 'Generate transcript')

    dot.edges([('d1', 'p1'), ('p1', 'd2'), ('d2', 'p2'), ('d3', 'p2'),
               ('p2', 'd5'), ('d2', 'p3'), ('d4', 'p3'), ('p3', 'd6'),
               ('d5', 'p4'), ('d6', 'p4'), ('p4', 'd7')])
    dot.render(name, cleanup=True)


def dfd_online_window(name='dfd_online_window', form='png',
                      label='Data flow diagram for windowed online speaker identification'):
    dot = gv.Digraph(comment=label, format=form)
    # Data nodes
    dot.attr('node', shape='box')
    dot.node('d1', 'Audio stream')
    dot.node('d2', 'Segments')
    dot.node('d3', 'Speaker model')
    dot.node('d5', 'Speaker hypotheses')
    dot.node('d6', 'Speaker prediction')

    # Process nodes
    dot.attr('node', shape='ellipse')
    dot.node('p1', 'Segment')
    dot.node('p2', 'Identify speakers')
    dot.node('p3', 'Identify speech')
    dot.node('p4', 'Generate transcript')

    dot.edges([('d1', 'p1'), ('p1', 'd2'), ('d2', 'p2'), ('d3', 'p2'),
               ('p2', 'd5'), ('d2', 'p3'), ('d4', 'p3'), ('p3', 'd6'),
               ('d5', 'p4'), ('d6', 'p4'), ('p4', 'd7')])
    dot.render(name, cleanup=True)


if __name__ == "__main__":
    dfd_labeling_single()
    dfd_labeling_all()
    dfd_baseline()
    dfd_offline()
    dfd_online()
