# Copyright 2025 The android_world Authors.
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

"""Utils for contacts operations using adb."""

import dataclasses
import logging
import re
import shlex
import time
from typing import Iterator

from android_env.proto import adb_pb2
from android_world.env import actuation
from android_world.env import adb_utils
from android_world.env import android_world_controller


_ACCOUNT_TYPE = "local"
_ACCOUNT_NAME = "local"
_PHONE_MIMETYPE = "vnd.android.cursor.item/phone_v2"
_NAME_MIMETYPE = "vnd.android.cursor.item/name"


def clean_phone_number(phone_number: str) -> str:
  """Removes all non-numeric characters from a phone number.

  Args:
    phone_number: The phone number to clean.

  Returns:
    The phone number with all non-numeric characters removed.
  """
  return re.sub(r"\D", "", phone_number)


def _shell_quote(value: str) -> str:
  return shlex.quote(str(value))


def _response_text(response) -> str:
  return response.generic.output.decode("utf-8", errors="ignore")


def _raw_contact_id_from_query(output: str) -> str | None:
  match = re.search(r"\b_id=(\d+)\b", output)
  return match.group(1) if match else None


def _add_contact_via_provider(
    name: str,
    phone_number: str,
    env: android_world_controller.AndroidWorldController,
) -> bool:
  """Adds a contact directly through ContactsProvider.

  AndroidWorld setup should not depend on the Google Contacts editor UI: on
  some real-device builds that activity crashes before rendering the Save
  button.  The provider path writes the same setup data without UI actuation.
  """
  raw_insert_command = (
      "content insert --uri content://com.android.contacts/raw_contacts"
      f" --bind account_type:s:{_shell_quote(_ACCOUNT_TYPE)}"
      f" --bind account_name:s:{_shell_quote(_ACCOUNT_NAME)}"
  )
  raw_response = adb_utils.issue_generic_request(["shell", raw_insert_command], env)
  if raw_response.status != adb_pb2.AdbResponse.Status.OK:
    logging.warning("ContactsProvider raw_contact insert failed; falling back.")
    return False

  query_command = (
      "content query --uri content://com.android.contacts/raw_contacts"
      ' --projection _id --sort "_id DESC"'
  )
  query_response = adb_utils.issue_generic_request(["shell", query_command], env)
  if query_response.status != adb_pb2.AdbResponse.Status.OK:
    logging.warning("ContactsProvider raw_contact query failed; falling back.")
    return False

  raw_contact_id = _raw_contact_id_from_query(_response_text(query_response))
  if raw_contact_id is None:
    logging.warning("ContactsProvider raw_contact id missing; falling back.")
    return False

  name_command = (
      "content insert --uri content://com.android.contacts/data"
      f" --bind raw_contact_id:i:{raw_contact_id}"
      f" --bind mimetype:s:{_shell_quote(_NAME_MIMETYPE)}"
      f" --bind data1:s:{_shell_quote(name)}"
  )
  name_response = adb_utils.issue_generic_request(["shell", name_command], env)
  if name_response.status != adb_pb2.AdbResponse.Status.OK:
    logging.warning("ContactsProvider name insert failed; falling back.")
    return False

  phone_command = (
      "content insert --uri content://com.android.contacts/data"
      f" --bind raw_contact_id:i:{raw_contact_id}"
      f" --bind mimetype:s:{_shell_quote(_PHONE_MIMETYPE)}"
      f" --bind data1:s:{_shell_quote(phone_number)}"
      " --bind data2:i:2"
  )
  phone_response = adb_utils.issue_generic_request(["shell", phone_command], env)
  if phone_response.status != adb_pb2.AdbResponse.Status.OK:
    logging.warning("ContactsProvider phone insert failed; falling back.")
    return False

  return True


def _add_contact_via_ui(
    name: str,
    phone_number: str,
    env: android_world_controller.AndroidWorldController,
    ui_delay_sec: float,
) -> None:
  """Adds a contact using the legacy UI flow."""
  intent_command = (
      "am start -a android.intent.action.INSERT -t"
      f' vnd.android.cursor.dir/contact -e name "{name}" -e phone'
      f" {phone_number}"
  )

  adb_command = ["shell", intent_command]
  adb_utils.issue_generic_request(adb_command, env)
  time.sleep(ui_delay_sec)
  actuation.find_and_click_element("SAVE", env)
  time.sleep(ui_delay_sec)
  adb_utils.press_back_button(env)
  time.sleep(ui_delay_sec)


def add_contact(
    name: str,
    phone_number: str,
    env: android_world_controller.AndroidWorldController,
    ui_delay_sec: float = 1.0,
):
  """Adds a contact with the specified name and phone number.

  This function sends an intent to the Android system to add a contact with
  the information pre-filled, clicks the "Save" button to create it, and then
  returns from the activity.

  Args:
    name: The name of the new contact
    phone_number: The phone number belonging to that contact.
    env: The android environment to add the contact to.
    ui_delay_sec: Delay between UI interactions. If this value is too low, the
      "save" button may be mis-clicked.
  """
  if _add_contact_via_provider(name, phone_number, env):
    return
  _add_contact_via_ui(name, phone_number, env, ui_delay_sec)


@dataclasses.dataclass(frozen=True)
class Contact:
  """Basic contact information."""
  name: str
  number: str


def list_contacts(
    env: android_world_controller.AndroidWorldController,
) -> list[Contact]:
  """Lists all contacts available in the Android environment.

  Args:
    env: Android environment to search for contacts.

  Returns:
    A list of all contact names and numbers present on the device.
  """
  modern_command = (
      "content query --uri content://com.android.contacts/data/phones"
      " --projection display_name:data1"
  )
  legacy_command = (
      "content query --uri content://contacts/phones/ --projection"
      " display_name:number"
  )

  def parse_modern(adb_output: str) -> Iterator[Contact]:
    for match in re.finditer(r"display_name=(.*?), data1=([^\n,]*)", adb_output):
      yield Contact(match.group(1), clean_phone_number(match.group(2)))

  def parse_legacy(adb_output: str) -> Iterator[Contact]:
    for match in re.finditer(r"display_name=(.*), number=(.*)", adb_output):
      yield Contact(match.group(1), clean_phone_number(match.group(2)))

  contacts = list(
      parse_modern(
          _response_text(
              adb_utils.issue_generic_request(["shell", modern_command], env)
          )
      )
  )
  if contacts:
    return contacts

  return list(
      parse_legacy(
          _response_text(
              adb_utils.issue_generic_request(["shell", legacy_command], env)
          )
      )
  )


def clear_contacts(env: android_world_controller.AndroidWorldController):
  """Clears all contacts on the device."""
  adb_utils.clear_app_data("com.android.providers.contacts", env)
