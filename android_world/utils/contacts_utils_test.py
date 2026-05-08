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

from unittest import mock

from absl.testing import absltest
from android_env import env_interface
from android_env.proto import adb_pb2
from android_world.env import actuation
from android_world.env import adb_utils
from android_world.utils import contacts_utils


@mock.patch.object(adb_utils, "issue_generic_request")
@mock.patch.object(actuation, "find_and_click_element")
class TestContactsUtils(absltest.TestCase):

  def test_add_contact(self, mock_click_element, mock_generic_request):
    """Test adding a contact through ContactsProvider."""
    mock_env = mock.create_autospec(env_interface.AndroidEnvInterface)
    ok = adb_pb2.AdbResponse()
    ok.status = adb_pb2.AdbResponse.Status.OK
    query = adb_pb2.AdbResponse()
    query.status = adb_pb2.AdbResponse.Status.OK
    query.generic.output = b"Row: 0 _id=7"
    mock_generic_request.side_effect = [ok, query, ok, ok]

    phone_number = "+123456789"
    name = "Emma Watson"
    contacts_utils.add_contact(name, phone_number, mock_env)

    mock_generic_request.assert_has_calls([
        mock.call([
            "shell",
            (
                "content insert --uri content://com.android.contacts/raw_contacts"
                " --bind account_type:s:local --bind account_name:s:local"
            ),
        ], mock_env),
        mock.call([
            "shell",
            (
                "content query --uri content://com.android.contacts/raw_contacts"
                ' --projection _id --sort "_id DESC"'
            ),
        ], mock_env),
        mock.call([
            "shell",
            (
                "content insert --uri content://com.android.contacts/data"
                " --bind raw_contact_id:i:7"
                " --bind mimetype:s:vnd.android.cursor.item/name"
                " --bind data1:s:'Emma Watson'"
            ),
        ], mock_env),
        mock.call([
            "shell",
            (
                "content insert --uri content://com.android.contacts/data"
                " --bind raw_contact_id:i:7"
                " --bind mimetype:s:vnd.android.cursor.item/phone_v2"
                " --bind data1:s:+123456789 --bind data2:i:2"
            ),
        ], mock_env),
    ])
    mock_click_element.assert_not_called()

  def test_add_contact_falls_back_to_ui(
      self, mock_click_element, mock_generic_request
  ):
    """Falls back to the legacy Contacts UI when provider insertion fails."""
    mock_env = mock.create_autospec(env_interface.AndroidEnvInterface)
    failed = adb_pb2.AdbResponse()
    failed.status = adb_pb2.AdbResponse.Status.INTERNAL_ERROR
    ok = adb_pb2.AdbResponse()
    ok.status = adb_pb2.AdbResponse.Status.OK
    mock_generic_request.side_effect = [failed, ok]

    phone_number = "+123456789"
    name = "Emma Watson"
    contacts_utils.add_contact(name, phone_number, mock_env)

    expected_adb_command = [
        "shell",
        (
            "am start -a android.intent.action.INSERT -t"
            f' vnd.android.cursor.dir/contact -e name "{name}" -e phone'
            f" {phone_number}"
        ),
    ]
    self.assertIn(mock.call(expected_adb_command, mock_env),
                  mock_generic_request.mock_calls)
    mock_click_element.assert_called_once_with("SAVE", mock_env)

  def test_list_contacts(self, unused_mock_click_element, mock_generic_request):
    """Test listing contacts from the modern phone data URI."""
    mock_env = mock.create_autospec(env_interface.AndroidEnvInterface)
    adb_response = adb_pb2.AdbResponse()
    adb_response.generic.output = """
      Row: 0 display_name=Jane Doe, data1=1 (234) 567-89
      Row: 0 display_name=Chen, data1=98765
    """.encode("utf-8")
    mock_generic_request.return_value = adb_response

    contacts = contacts_utils.list_contacts(mock_env)

    self.assertEqual(
        contacts,
        [
            contacts_utils.Contact("Jane Doe", "123456789"),
            contacts_utils.Contact("Chen", "98765"),
        ],
    )

    expected_adb_command = [
        "shell",
        (
            "content query --uri content://com.android.contacts/data/phones"
            " --projection display_name:data1"
        ),
    ]
    mock_generic_request.assert_called_once_with(expected_adb_command, mock_env)

  def test_list_contacts_falls_back_to_legacy_uri(
      self, unused_mock_click_element, mock_generic_request
  ):
    """Falls back to the legacy phone URI when the modern URI is empty."""
    mock_env = mock.create_autospec(env_interface.AndroidEnvInterface)
    empty = adb_pb2.AdbResponse()
    empty.generic.output = b"No result found."
    legacy = adb_pb2.AdbResponse()
    legacy.generic.output = """
      Row: 0 display_name=Jane Doe, number=1 (234) 567-89
      Row: 0 display_name=Chen, number=98765
    """.encode("utf-8")
    mock_generic_request.side_effect = [empty, legacy]

    contacts = contacts_utils.list_contacts(mock_env)

    self.assertEqual(
        contacts,
        [
            contacts_utils.Contact("Jane Doe", "123456789"),
            contacts_utils.Contact("Chen", "98765"),
        ],
    )

  def test_clear_contacts(
      self, unused_mock_click_element, mock_generic_request
  ):
    """Test clearing all contacts."""
    mock_env = mock.create_autospec(env_interface.AndroidEnvInterface)

    contacts_utils.clear_contacts(mock_env)

    # Construct the expected adb command
    expected_adb_command = [
        "shell",
        "pm",
        "clear",
        "com.android.providers.contacts",
    ]

    # Assert that the correct adb command was issued
    mock_generic_request.assert_called_once_with(expected_adb_command, mock_env)


if __name__ == "__main__":
  absltest.main()
