#!/usr/bin/env bash
while [ 1 ]
do 
	git add ./models
	git commit -m "Adding test"
	git push origin sharing
	echo "Waiting 2 minutes..."
	sleep 2m
done
