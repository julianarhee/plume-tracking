#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File           : pull_google_sheet.py
Created        : 2022/11/17 15:56:32
Project        : /Users/julianarhee/Repositories/plume-tracking
Author         : jyr
Email          : juliana.rhee@gmail.com
Last Modified  : 

Notes: 
Step-by-step instructions from:
https://towardsdatascience.com/how-to-import-google-sheets-data-into-a-pandas-dataframe-using-googles-api-v4-2020-f50e84ea4530

Using existing credentials.json from:
https://github.com/rutalaboratory/edge-tracking

'''
import pickle
import os.path
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

import pandas as pd

def gsheet_api_check(SCOPES, parent_dir = '/Users/julianarhee/Repositories/plume-tracking'):
    #print(os.listdir(parent_dir))
    creds = None
    if os.path.exists(os.path.join(parent_dir, 'token.pickle')):
        with open(os.path.join(parent_dir, 'token.pickle'), 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                os.path.join(parent_dir, 'credentials.json'), SCOPES)
            creds = flow.run_local_server(port=0)
        with open(os.path.join(parent_dir, 'token.pickle'), 'wb') as token:
            pickle.dump(creds, token)
    return creds

from googleapiclient.discovery import build
def pull_sheet_data(SCOPES,SPREADSHEET_ID,DATA_TO_PULL):
    creds = gsheet_api_check(SCOPES)
    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()
    result = sheet.values().get(
        spreadsheetId=SPREADSHEET_ID,
        range=DATA_TO_PULL).execute()
    values = result.get('values', [])
    
    if not values:
        print('No data found.')
    else:
        rows = sheet.values().get(spreadsheetId=SPREADSHEET_ID,
                                  range=DATA_TO_PULL).execute()
        data = rows.get('values')
        print("COMPLETE: Data copied")
        return data

# custom funcs for us

def gsheet_to_dataframe(SPREADSHEET_ID, DATA_TO_PULL='Sheet1',
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']):
    #SPREADSHEET_ID = '1K_SkaT3JUA2Ik8uiwB6kJMwHnd935bZvZB4If0zh8rY'
    #Pulls data from the entire spreadsheet tab.
    #DATA_TO_PULL = 'Sheet1'
    data = pull_sheet_data(SCOPES,SPREADSHEET_ID,DATA_TO_PULL)
    df = pd.DataFrame(data[1:], columns=data[0])
    #print(df.head())
    return df

def get_sheet_id(experiment):
    if experiment == '0-degree':
        sheet_id = '1K_SkaT3JUA2Ik8uiwB6kJMwHnd935bZvZB4If0zh8rY'
    elif experiment == '15-degree':
        sheet_id = '1qCrV96jUo24lpZ7-k2-B9RWG5RSQDSgnSn-sFgjS7ys'
    elif experiment == '45-degree':
        sheet_id = '15mE8k1Z9PN3_xhQH6mz1AEIyspjlfg5KPkd1aNLs9TM'
    elif experiment in ['T-plume', 't-plume', 'tplume', '90-degree']:
        sheet_id = '14r0TgRUhohZtw2GQgirUseBWXK8NPbyqPzPvAtND7Gs'
    elif experiment in ['constant_gradient', 'constant_vs_gradient']:
        sheet_id = '1Is1t3UtMAycrvpSMvEf6j2Gpc4b5jkEdm7yTIEAxfw8'
    else:
        sheet_id = None

    return sheet_id

def get_info_from_gsheet(experiment):
    '''
    For experiment name (only a subset specified), pull google sheet
    info into dataframe.

    Arguments:
        experiment -- _description_
        tab_name -- _description_

    Returns:
        _description_
    '''

    SHEET_ID = get_sheet_id(experiment)
    df = gsheet_to_dataframe(SHEET_ID)
    return df

    
def main():
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
    SPREADSHEET_ID = '1K_SkaT3JUA2Ik8uiwB6kJMwHnd935bZvZB4If0zh8rY'
    #Pulls data from the entire spreadsheet tab.
    DATA_TO_PULL = 'Sheet1'
    data = pull_sheet_data(SCOPES,SPREADSHEET_ID,DATA_TO_PULL)
    df = pd.DataFrame(data[1:], columns=data[0])
    print(df.head())


if __name__ == '__main__':
    main() 