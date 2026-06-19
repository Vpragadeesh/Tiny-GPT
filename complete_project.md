Project Path: Tiny-GPT

Source Tree:

```txt
Tiny-GPT
├── LICENSE
├── README.md
├── complete_project.md
├── data
│   └── meta.txt
├── deepspeed.md
├── ds_config.active.json
├── ds_config.json
├── hf_cache
├── main.py
├── main_deepspeed.py
├── prepare_data.py
├── push_to_hf.py
├── run.py
└── train_deepspeed.sh

```

`LICENSE`:

```
                    GNU AFFERO GENERAL PUBLIC LICENSE
                       Version 3, 19 November 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

                            Preamble

  The GNU Affero General Public License is a free, copyleft license for
software and other kinds of works, specifically designed to ensure
cooperation with the community in the case of network server software.

  The licenses for most software and other practical works are designed
to take away your freedom to share and change the works.  By contrast,
our General Public Licenses are intended to guarantee your freedom to
share and change all versions of a program--to make sure it remains free
software for all its users.

  When we speak of free software, we are referring to freedom, not
price.  Our General Public Licenses are designed to make sure that you
have the freedom to distribute copies of free software (and charge for
them if you wish), that you receive source code or can get it if you
want it, that you can change the software or use pieces of it in new
free programs, and that you know you can do these things.

  Developers that use our General Public Licenses protect your rights
with two steps: (1) assert copyright on the software, and (2) offer
you this License which gives you legal permission to copy, distribute
and/or modify the software.

  A secondary benefit of defending all users' freedom is that
improvements made in alternate versions of the program, if they
receive widespread use, become available for other developers to
incorporate.  Many developers of free software are heartened and
encouraged by the resulting cooperation.  However, in the case of
software used on network servers, this result may fail to come about.
The GNU General Public License permits making a modified version and
letting the public access it on a server without ever releasing its
source code to the public.

  The GNU Affero General Public License is designed specifically to
ensure that, in such cases, the modified source code becomes available
to the community.  It requires the operator of a network server to
provide the source code of the modified version running there to the
users of that server.  Therefore, public use of a modified version, on
a publicly accessible server, gives the public access to the source
code of the modified version.

  An older license, called the Affero General Public License and
published by Affero, was designed to accomplish similar goals.  This is
a different license, not a version of the Affero GPL, but Affero has
released a new version of the Affero GPL which permits relicensing under
this license.

  The precise terms and conditions for copying, distribution and
modification follow.

                       TERMS AND CONDITIONS

  0. Definitions.

  "This License" refers to version 3 of the GNU Affero General Public License.

  "Copyright" also means copyright-like laws that apply to other kinds of
works, such as semiconductor masks.

  "The Program" refers to any copyrightable work licensed under this
License.  Each licensee is addressed as "you".  "Licensees" and
"recipients" may be individuals or organizations.

  To "modify" a work means to copy from or adapt all or part of the work
in a fashion requiring copyright permission, other than the making of an
exact copy.  The resulting work is called a "modified version" of the
earlier work or a work "based on" the earlier work.

  A "covered work" means either the unmodified Program or a work based
on the Program.

  To "propagate" a work means to do anything with it that, without
permission, would make you directly or secondarily liable for
infringement under applicable copyright law, except executing it on a
computer or modifying a private copy.  Propagation includes copying,
distribution (with or without modification), making available to the
public, and in some countries other activities as well.

  To "convey" a work means any kind of propagation that enables other
parties to make or receive copies.  Mere interaction with a user through
a computer network, with no transfer of a copy, is not conveying.

  An interactive user interface displays "Appropriate Legal Notices"
to the extent that it includes a convenient and prominently visible
feature that (1) displays an appropriate copyright notice, and (2)
tells the user that there is no warranty for the work (except to the
extent that warranties are provided), that licensees may convey the
work under this License, and how to view a copy of this License.  If
the interface presents a list of user commands or options, such as a
menu, a prominent item in the list meets this criterion.

  1. Source Code.

  The "source code" for a work means the preferred form of the work
for making modifications to it.  "Object code" means any non-source
form of a work.

  A "Standard Interface" means an interface that either is an official
standard defined by a recognized standards body, or, in the case of
interfaces specified for a particular programming language, one that
is widely used among developers working in that language.

  The "System Libraries" of an executable work include anything, other
than the work as a whole, that (a) is included in the normal form of
packaging a Major Component, but which is not part of that Major
Component, and (b) serves only to enable use of the work with that
Major Component, or to implement a Standard Interface for which an
implementation is available to the public in source code form.  A
"Major Component", in this context, means a major essential component
(kernel, window system, and so on) of the specific operating system
(if any) on which the executable work runs, or a compiler used to
produce the work, or an object code interpreter used to run it.

  The "Corresponding Source" for a work in object code form means all
the source code needed to generate, install, and (for an executable
work) run the object code and to modify the work, including scripts to
control those activities.  However, it does not include the work's
System Libraries, or general-purpose tools or generally available free
programs which are used unmodified in performing those activities but
which are not part of the work.  For example, Corresponding Source
includes interface definition files associated with source files for
the work, and the source code for shared libraries and dynamically
linked subprograms that the work is specifically designed to require,
such as by intimate data communication or control flow between those
subprograms and other parts of the work.

  The Corresponding Source need not include anything that users
can regenerate automatically from other parts of the Corresponding
Source.

  The Corresponding Source for a work in source code form is that
same work.

  2. Basic Permissions.

  All rights granted under this License are granted for the term of
copyright on the Program, and are irrevocable provided the stated
conditions are met.  This License explicitly affirms your unlimited
permission to run the unmodified Program.  The output from running a
covered work is covered by this License only if the output, given its
content, constitutes a covered work.  This License acknowledges your
rights of fair use or other equivalent, as provided by copyright law.

  You may make, run and propagate covered works that you do not
convey, without conditions so long as your license otherwise remains
in force.  You may convey covered works to others for the sole purpose
of having them make modifications exclusively for you, or provide you
with facilities for running those works, provided that you comply with
the terms of this License in conveying all material for which you do
not control copyright.  Those thus making or running the covered works
for you must do so exclusively on your behalf, under your direction
and control, on terms that prohibit them from making any copies of
your copyrighted material outside their relationship with you.

  Conveying under any other circumstances is permitted solely under
the conditions stated below.  Sublicensing is not allowed; section 10
makes it unnecessary.

  3. Protecting Users' Legal Rights From Anti-Circumvention Law.

  No covered work shall be deemed part of an effective technological
measure under any applicable law fulfilling obligations under article
11 of the WIPO copyright treaty adopted on 20 December 1996, or
similar laws prohibiting or restricting circumvention of such
measures.

  When you convey a covered work, you waive any legal power to forbid
circumvention of technological measures to the extent such circumvention
is effected by exercising rights under this License with respect to
the covered work, and you disclaim any intention to limit operation or
modification of the work as a means of enforcing, against the work's
users, your or third parties' legal rights to forbid circumvention of
technological measures.

  4. Conveying Verbatim Copies.

  You may convey verbatim copies of the Program's source code as you
receive it, in any medium, provided that you conspicuously and
appropriately publish on each copy an appropriate copyright notice;
keep intact all notices stating that this License and any
non-permissive terms added in accord with section 7 apply to the code;
keep intact all notices of the absence of any warranty; and give all
recipients a copy of this License along with the Program.

  You may charge any price or no price for each copy that you convey,
and you may offer support or warranty protection for a fee.

  5. Conveying Modified Source Versions.

  You may convey a work based on the Program, or the modifications to
produce it from the Program, in the form of source code under the
terms of section 4, provided that you also meet all of these conditions:

    a) The work must carry prominent notices stating that you modified
    it, and giving a relevant date.

    b) The work must carry prominent notices stating that it is
    released under this License and any conditions added under section
    7.  This requirement modifies the requirement in section 4 to
    "keep intact all notices".

    c) You must license the entire work, as a whole, under this
    License to anyone who comes into possession of a copy.  This
    License will therefore apply, along with any applicable section 7
    additional terms, to the whole of the work, and all its parts,
    regardless of how they are packaged.  This License gives no
    permission to license the work in any other way, but it does not
    invalidate such permission if you have separately received it.

    d) If the work has interactive user interfaces, each must display
    Appropriate Legal Notices; however, if the Program has interactive
    interfaces that do not display Appropriate Legal Notices, your
    work need not make them do so.

  A compilation of a covered work with other separate and independent
works, which are not by their nature extensions of the covered work,
and which are not combined with it such as to form a larger program,
in or on a volume of a storage or distribution medium, is called an
"aggregate" if the compilation and its resulting copyright are not
used to limit the access or legal rights of the compilation's users
beyond what the individual works permit.  Inclusion of a covered work
in an aggregate does not cause this License to apply to the other
parts of the aggregate.

  6. Conveying Non-Source Forms.

  You may convey a covered work in object code form under the terms
of sections 4 and 5, provided that you also convey the
machine-readable Corresponding Source under the terms of this License,
in one of these ways:

    a) Convey the object code in, or embodied in, a physical product
    (including a physical distribution medium), accompanied by the
    Corresponding Source fixed on a durable physical medium
    customarily used for software interchange.

    b) Convey the object code in, or embodied in, a physical product
    (including a physical distribution medium), accompanied by a
    written offer, valid for at least three years and valid for as
    long as you offer spare parts or customer support for that product
    model, to give anyone who possesses the object code either (1) a
    copy of the Corresponding Source for all the software in the
    product that is covered by this License, on a durable physical
    medium customarily used for software interchange, for a price no
    more than your reasonable cost of physically performing this
    conveying of source, or (2) access to copy the
    Corresponding Source from a network server at no charge.

    c) Convey individual copies of the object code with a copy of the
    written offer to provide the Corresponding Source.  This
    alternative is allowed only occasionally and noncommercially, and
    only if you received the object code with such an offer, in accord
    with subsection 6b.

    d) Convey the object code by offering access from a designated
    place (gratis or for a charge), and offer equivalent access to the
    Corresponding Source in the same way through the same place at no
    further charge.  You need not require recipients to copy the
    Corresponding Source along with the object code.  If the place to
    copy the object code is a network server, the Corresponding Source
    may be on a different server (operated by you or a third party)
    that supports equivalent copying facilities, provided you maintain
    clear directions next to the object code saying where to find the
    Corresponding Source.  Regardless of what server hosts the
    Corresponding Source, you remain obligated to ensure that it is
    available for as long as needed to satisfy these requirements.

    e) Convey the object code using peer-to-peer transmission, provided
    you inform other peers where the object code and Corresponding
    Source of the work are being offered to the general public at no
    charge under subsection 6d.

  A separable portion of the object code, whose source code is excluded
from the Corresponding Source as a System Library, need not be
included in conveying the object code work.

  A "User Product" is either (1) a "consumer product", which means any
tangible personal property which is normally used for personal, family,
or household purposes, or (2) anything designed or sold for incorporation
into a dwelling.  In determining whether a product is a consumer product,
doubtful cases shall be resolved in favor of coverage.  For a particular
product received by a particular user, "normally used" refers to a
typical or common use of that class of product, regardless of the status
of the particular user or of the way in which the particular user
actually uses, or expects or is expected to use, the product.  A product
is a consumer product regardless of whether the product has substantial
commercial, industrial or non-consumer uses, unless such uses represent
the only significant mode of use of the product.

  "Installation Information" for a User Product means any methods,
procedures, authorization keys, or other information required to install
and execute modified versions of a covered work in that User Product from
a modified version of its Corresponding Source.  The information must
suffice to ensure that the continued functioning of the modified object
code is in no case prevented or interfered with solely because
modification has been made.

  If you convey an object code work under this section in, or with, or
specifically for use in, a User Product, and the conveying occurs as
part of a transaction in which the right of possession and use of the
User Product is transferred to the recipient in perpetuity or for a
fixed term (regardless of how the transaction is characterized), the
Corresponding Source conveyed under this section must be accompanied
by the Installation Information.  But this requirement does not apply
if neither you nor any third party retains the ability to install
modified object code on the User Product (for example, the work has
been installed in ROM).

  The requirement to provide Installation Information does not include a
requirement to continue to provide support service, warranty, or updates
for a work that has been modified or installed by the recipient, or for
the User Product in which it has been modified or installed.  Access to a
network may be denied when the modification itself materially and
adversely affects the operation of the network or violates the rules and
protocols for communication across the network.

  Corresponding Source conveyed, and Installation Information provided,
in accord with this section must be in a format that is publicly
documented (and with an implementation available to the public in
source code form), and must require no special password or key for
unpacking, reading or copying.

  7. Additional Terms.

  "Additional permissions" are terms that supplement the terms of this
License by making exceptions from one or more of its conditions.
Additional permissions that are applicable to the entire Program shall
be treated as though they were included in this License, to the extent
that they are valid under applicable law.  If additional permissions
apply only to part of the Program, that part may be used separately
under those permissions, but the entire Program remains governed by
this License without regard to the additional permissions.

  When you convey a copy of a covered work, you may at your option
remove any additional permissions from that copy, or from any part of
it.  (Additional permissions may be written to require their own
removal in certain cases when you modify the work.)  You may place
additional permissions on material, added by you to a covered work,
for which you have or can give appropriate copyright permission.

  Notwithstanding any other provision of this License, for material you
add to a covered work, you may (if authorized by the copyright holders of
that material) supplement the terms of this License with terms:

    a) Disclaiming warranty or limiting liability differently from the
    terms of sections 15 and 16 of this License; or

    b) Requiring preservation of specified reasonable legal notices or
    author attributions in that material or in the Appropriate Legal
    Notices displayed by works containing it; or

    c) Prohibiting misrepresentation of the origin of that material, or
    requiring that modified versions of such material be marked in
    reasonable ways as different from the original version; or

    d) Limiting the use for publicity purposes of names of licensors or
    authors of the material; or

    e) Declining to grant rights under trademark law for use of some
    trade names, trademarks, or service marks; or

    f) Requiring indemnification of licensors and authors of that
    material by anyone who conveys the material (or modified versions of
    it) with contractual assumptions of liability to the recipient, for
    any liability that these contractual assumptions directly impose on
    those licensors and authors.

  All other non-permissive additional terms are considered "further
restrictions" within the meaning of section 10.  If the Program as you
received it, or any part of it, contains a notice stating that it is
governed by this License along with a term that is a further
restriction, you may remove that term.  If a license document contains
a further restriction but permits relicensing or conveying under this
License, you may add to a covered work material governed by the terms
of that license document, provided that the further restriction does
not survive such relicensing or conveying.

  If you add terms to a covered work in accord with this section, you
must place, in the relevant source files, a statement of the
additional terms that apply to those files, or a notice indicating
where to find the applicable terms.

  Additional terms, permissive or non-permissive, may be stated in the
form of a separately written license, or stated as exceptions;
the above requirements apply either way.

  8. Termination.

  You may not propagate or modify a covered work except as expressly
provided under this License.  Any attempt otherwise to propagate or
modify it is void, and will automatically terminate your rights under
this License (including any patent licenses granted under the third
paragraph of section 11).

  However, if you cease all violation of this License, then your
license from a particular copyright holder is reinstated (a)
provisionally, unless and until the copyright holder explicitly and
finally terminates your license, and (b) permanently, if the copyright
holder fails to notify you of the violation by some reasonable means
prior to 60 days after the cessation.

  Moreover, your license from a particular copyright holder is
reinstated permanently if the copyright holder notifies you of the
violation by some reasonable means, this is the first time you have
received notice of violation of this License (for any work) from that
copyright holder, and you cure the violation prior to 30 days after
your receipt of the notice.

  Termination of your rights under this section does not terminate the
licenses of parties who have received copies or rights from you under
this License.  If your rights have been terminated and not permanently
reinstated, you do not qualify to receive new licenses for the same
material under section 10.

  9. Acceptance Not Required for Having Copies.

  You are not required to accept this License in order to receive or
run a copy of the Program.  Ancillary propagation of a covered work
occurring solely as a consequence of using peer-to-peer transmission
to receive a copy likewise does not require acceptance.  However,
nothing other than this License grants you permission to propagate or
modify any covered work.  These actions infringe copyright if you do
not accept this License.  Therefore, by modifying or propagating a
covered work, you indicate your acceptance of this License to do so.

  10. Automatic Licensing of Downstream Recipients.

  Each time you convey a covered work, the recipient automatically
receives a license from the original licensors, to run, modify and
propagate that work, subject to this License.  You are not responsible
for enforcing compliance by third parties with this License.

  An "entity transaction" is a transaction transferring control of an
organization, or substantially all assets of one, or subdividing an
organization, or merging organizations.  If propagation of a covered
work results from an entity transaction, each party to that
transaction who receives a copy of the work also receives whatever
licenses to the work the party's predecessor in interest had or could
give under the previous paragraph, plus a right to possession of the
Corresponding Source of the work from the predecessor in interest, if
the predecessor has it or can get it with reasonable efforts.

  You may not impose any further restrictions on the exercise of the
rights granted or affirmed under this License.  For example, you may
not impose a license fee, royalty, or other charge for exercise of
rights granted under this License, and you may not initiate litigation
(including a cross-claim or counterclaim in a lawsuit) alleging that
any patent claim is infringed by making, using, selling, offering for
sale, or importing the Program or any portion of it.

  11. Patents.

  A "contributor" is a copyright holder who authorizes use under this
License of the Program or a work on which the Program is based.  The
work thus licensed is called the contributor's "contributor version".

  A contributor's "essential patent claims" are all patent claims
owned or controlled by the contributor, whether already acquired or
hereafter acquired, that would be infringed by some manner, permitted
by this License, of making, using, or selling its contributor version,
but do not include claims that would be infringed only as a
consequence of further modification of the contributor version.  For
purposes of this definition, "control" includes the right to grant
patent sublicenses in a manner consistent with the requirements of
this License.

  Each contributor grants you a non-exclusive, worldwide, royalty-free
patent license under the contributor's essential patent claims, to
make, use, sell, offer for sale, import and otherwise run, modify and
propagate the contents of its contributor version.

  In the following three paragraphs, a "patent license" is any express
agreement or commitment, however denominated, not to enforce a patent
(such as an express permission to practice a patent or covenant not to
sue for patent infringement).  To "grant" such a patent license to a
party means to make such an agreement or commitment not to enforce a
patent against the party.

  If you convey a covered work, knowingly relying on a patent license,
and the Corresponding Source of the work is not available for anyone
to copy, free of charge and under the terms of this License, through a
publicly available network server or other readily accessible means,
then you must either (1) cause the Corresponding Source to be so
available, or (2) arrange to deprive yourself of the benefit of the
patent license for this particular work, or (3) arrange, in a manner
consistent with the requirements of this License, to extend the patent
license to downstream recipients.  "Knowingly relying" means you have
actual knowledge that, but for the patent license, your conveying the
covered work in a country, or your recipient's use of the covered work
in a country, would infringe one or more identifiable patents in that
country that you have reason to believe are valid.

  If, pursuant to or in connection with a single transaction or
arrangement, you convey, or propagate by procuring conveyance of, a
covered work, and grant a patent license to some of the parties
receiving the covered work authorizing them to use, propagate, modify
or convey a specific copy of the covered work, then the patent license
you grant is automatically extended to all recipients of the covered
work and works based on it.

  A patent license is "discriminatory" if it does not include within
the scope of its coverage, prohibits the exercise of, or is
conditioned on the non-exercise of one or more of the rights that are
specifically granted under this License.  You may not convey a covered
work if you are a party to an arrangement with a third party that is
in the business of distributing software, under which you make payment
to the third party based on the extent of your activity of conveying
the work, and under which the third party grants, to any of the
parties who would receive the covered work from you, a discriminatory
patent license (a) in connection with copies of the covered work
conveyed by you (or copies made from those copies), or (b) primarily
for and in connection with specific products or compilations that
contain the covered work, unless you entered into that arrangement,
or that patent license was granted, prior to 28 March 2007.

  Nothing in this License shall be construed as excluding or limiting
any implied license or other defenses to infringement that may
otherwise be available to you under applicable patent law.

  12. No Surrender of Others' Freedom.

  If conditions are imposed on you (whether by court order, agreement or
otherwise) that contradict the conditions of this License, they do not
excuse you from the conditions of this License.  If you cannot convey a
covered work so as to satisfy simultaneously your obligations under this
License and any other pertinent obligations, then as a consequence you may
not convey it at all.  For example, if you agree to terms that obligate you
to collect a royalty for further conveying from those to whom you convey
the Program, the only way you could satisfy both those terms and this
License would be to refrain entirely from conveying the Program.

  13. Remote Network Interaction; Use with the GNU General Public License.

  Notwithstanding any other provision of this License, if you modify the
Program, your modified version must prominently offer all users
interacting with it remotely through a computer network (if your version
supports such interaction) an opportunity to receive the Corresponding
Source of your version by providing access to the Corresponding Source
from a network server at no charge, through some standard or customary
means of facilitating copying of software.  This Corresponding Source
shall include the Corresponding Source for any work covered by version 3
of the GNU General Public License that is incorporated pursuant to the
following paragraph.

  Notwithstanding any other provision of this License, you have
permission to link or combine any covered work with a work licensed
under version 3 of the GNU General Public License into a single
combined work, and to convey the resulting work.  The terms of this
License will continue to apply to the part which is the covered work,
but the work with which it is combined will remain governed by version
3 of the GNU General Public License.

  14. Revised Versions of this License.

  The Free Software Foundation may publish revised and/or new versions of
the GNU Affero General Public License from time to time.  Such new versions
will be similar in spirit to the present version, but may differ in detail to
address new problems or concerns.

  Each version is given a distinguishing version number.  If the
Program specifies that a certain numbered version of the GNU Affero General
Public License "or any later version" applies to it, you have the
option of following the terms and conditions either of that numbered
version or of any later version published by the Free Software
Foundation.  If the Program does not specify a version number of the
GNU Affero General Public License, you may choose any version ever published
by the Free Software Foundation.

  If the Program specifies that a proxy can decide which future
versions of the GNU Affero General Public License can be used, that proxy's
public statement of acceptance of a version permanently authorizes you
to choose that version for the Program.

  Later license versions may give you additional or different
permissions.  However, no additional obligations are imposed on any
author or copyright holder as a result of your choosing to follow a
later version.

  15. Disclaimer of Warranty.

  THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY
APPLICABLE LAW.  EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT
HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY
OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM
IS WITH YOU.  SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF
ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

  16. Limitation of Liability.

  IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING
WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MODIFIES AND/OR CONVEYS
THE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY
GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE
USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF
DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD
PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS),
EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF
SUCH DAMAGES.

  17. Interpretation of Sections 15 and 16.

  If the disclaimer of warranty and limitation of liability provided
above cannot be given local legal effect according to their terms,
reviewing courts shall apply local law that most closely approximates
an absolute waiver of all civil liability in connection with the
Program, unless a warranty or assumption of liability accompanies a
copy of the Program in return for a fee.

                     END OF TERMS AND CONDITIONS

            How to Apply These Terms to Your New Programs

  If you develop a new program, and you want it to be of the greatest
possible use to the public, the best way to achieve this is to make it
free software which everyone can redistribute and change under these terms.

  To do so, attach the following notices to the program.  It is safest
to attach them to the start of each source file to most effectively
state the exclusion of warranty; and each file should have at least
the "copyright" line and a pointer to where the full notice is found.

    <one line to give the program's name and a brief idea of what it does.>
    Copyright (C) <year>  <name of author>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Also add information on how to contact you by electronic and paper mail.

  If your software can interact with users remotely through a computer
network, you should also make sure that it provides a way for users to
get its source.  For example, if your program is a web application, its
interface could display a "Source" link that leads users to an archive
of the code.  There are many ways you could offer source, and different
solutions will be better for different programs; see section 13 for the
specific requirements.

  You should also get your employer (if you work as a programmer) or school,
if any, to sign a "copyright disclaimer" for the program, if necessary.
For more information on this, and how to apply and follow the GNU AGPL, see
<https://www.gnu.org/licenses/>.

```
`README.md`:

```md
# Tiny-GPT: 0.5B MoE Language Model

A clean, efficient implementation of a **Mixture-of-Experts GPT** that fits on modest GPUs (4GB VRAM) while training on large datasets.

## 🎯 Main Goal
**Generate proper English text** - not gibberish!

## 📊 Quick Stats

| Metric | Value |
|--------|-------|
| **Model Size** | 0.5B parameters (520M) |
| **Active per Token** | 180M parameters (via MoE routing) |
| **Architecture** | 12 Transformer layers, 8 experts/layer, top-2 routing |
| **Training Data** | WikiText-103 (103M tokens, ~500MB) |
| **GPU Memory** | 0.97 GiB (model weights only) |
| **Training Time** | ~10-20 hours on RTX 2050 (10k steps) |
| **Tokenizer** | GPT-2 BPE (50,257 vocab via tiktoken) |

## 🚀 Quick Start

### 1. Prepare Dataset
```bash
python prepare_data.py
```
Downloads WikiText-103 and tokenizes to memory-mapped binary files (~500MB).
This is a one-time operation that takes **10-30 minutes**.

### 2. Train
```bash
python main.py
```
Starts training from scratch with:
- **Learning rate**: 1.5e-4 (lowered for stability)
- **Warmup**: 500 steps (better convergence)
- **Total steps**: 10,000 (more thorough training)
- **Batch size**: 16 (gradient accumulation of 2x8)

Training progress shows in real-time via rich progress bar.

### 3. Generate Text
```bash
python run.py
```

## 🤗 Use Hugging Face Hub (instead of local/GitHub checkpoints)

### 1. Upload checkpoints to HF Hub
```bash
pip install huggingface_hub
export HF_TOKEN=your_hf_token
python push_to_hf.py --repo-id yourname/Tiny-GPT
```

This uploads:
- `checkpoints/best.pt` → `best.pt`
- `checkpoints/latest.pt` → `latest.pt` (if present)

### 2. Run inference directly from HF Hub
```bash
python run.py --hf-repo yourname/Tiny-GPT --prompt "The future of AI is"
```

Optional flags:
- `--hf-filename best.pt`
- `--hf-revision main`
- `--hf-token <token>` (or use `HF_TOKEN` env var)

## 📁 File Structure

```
Tiny-GPT/
├── main.py                  # Training script
├── run.py                   # Inference script (NEW)
├── prepare_data.py          # Dataset preparation
├── mini_gpt.py              # Deprecated v1 (reference only)
├── reset_training.sh        # Clean old checkpoints
├── wait_for_dataset.sh      # Monitor data preparation
│
├── data/
│   ├── train.bin            # ~1.8M examples → ~80M tokens
│   ├── val.bin              # ~3.7k examples → ~1.7M tokens
│   ├── test.bin             # ~4.3k examples → ~2.0M tokens
│   └── meta.txt             # Metadata
│
└── checkpoints/
    ├── latest.pt            # Most recent checkpoint
    └── best.pt              # Best validation loss checkpoint
```

## 🔧 Configuration

All hyperparameters are defined in `main.py`:

```python
BLOCK_SIZE    = 128              # Context window
EMBED_DIM     = 768              # Model width
NUM_LAYERS    = 12               # Transformer blocks
NUM_EXPERTS   = 8                # Experts per MoE layer
TOP_K         = 2                # Experts used per token
LR            = 1.5e-4           # Learning rate (adjusted)
WARMUP_STEPS  = 500              # Warmup schedule
MAX_ITERS     = 10000            # Total training steps
GRAD_CLIP     = 1.0              # Gradient clipping
```

## 📈 Expected Training Progress

**With fixed hyperparameters (new):**
- **Step 1**: Loss ~8.0
- **Step 500**: Loss ~6.5-7.0
- **Step 2500**: Loss ~4.5-5.0
- **Step 5000**: Loss ~3.8-4.2
- **Step 10000**: Loss ~3.5-3.8

**Quality indicator:** Model starts generating coherent English by step 2000+

## 💡 What Changed?

### Before (Broken)
```
Learning Rate: 3e-4 (too high)
Warmup: 200 steps (insufficient)
Auto-resume: Enabled (got stuck in NaN)
Trainer Loss: DIVERGES TO NAN
Output: "hi defencesaternal Thirty shows allowanceBad Leh..."  ❌
```

### After (Fixed)
```
Learning Rate: 1.5e-4 (stable)
Warmup: 500 steps (better convergence)
Auto-resume: Disabled (start fresh)
Training Loss: SMOOTH CONVERGENCE
Output: "The history of the universe began with the Big Bang..."  ✓
```

## 🧠 Model Architecture

```
Input Tokens
    ↓
Embedding + Positional Encoding (768-dim)
    ↓
[x12 Transformer Blocks]
  ├─ Multi-Head Attention (12 heads)
  │  └─ Output: 768-dim
  └─ Mixture-of-Experts Layer
     ├─ 8 Expert FFNs (768→3072→768)
     ├─ Router: Selects top-2 experts per token
     └─ Load-balancing auxiliary loss
    ↓
Layer Norm
    ↓
Output Linear → Logits (50,257)
    ↓
Cross-Entropy Loss
```

**Memory Trick:** The CPUOffloadAdamW optimizer keeps fp32 master weights + momentum/variance on CPU RAM to save GPU VRAM:
- GPU: fp16 model weights + fp16 gradients (~1 GB)
- CPU: fp32 master weights + fp32 m/v (~4 GB)

## 🎮 Using `run.py`

### Interactive Mode (Default)
```bash
python run.py
```
Type prompts and press Enter. Commands:
- `/temp 0.8` - Set temperature (higher = more random)
- `/len 150` - Set max tokens
- `/topk 40` - Enable top-k sampling
- `/topp 0.9` - Set nucleus sampling threshold
- `quit` - Exit

### Single Prompt
```bash
python run.py --prompt "The future of AI is"
```

### Batch from File
```bash
python run.py --prompts prompts.txt  # One prompt per line
```

### Custom Checkpoint
```bash
python run.py --checkpoint checkpoints/best.pt
```

### Full Options
```bash
python run.py --help
```

## 🔍 Monitoring Training

The training loop shows:
```
Step  5000  │  Train 4.23  │  Val 4.45  │  LR 0.000097
```

**Healthy indicators:**
- ✓ Train loss smoothly decreases
- ✓ Val loss follows trend
- ✓ No NaN values
- ✓ Learning rate schedule works
- ✓ No gradient clipping (or occasional, < 10% of steps)

**Red flags:**
- ❌ Loss jumps/oscillates wildly
- ❌ NaN values appear
- ❌ Val loss stops improving (need more data or different HP)
- ❌ Constant gradient clipping (reduce LR)

## 📊 Checkpointing

Saved automatically every 500 steps:
- **`latest.pt`**: Most recent checkpoint (always usable)
- **`best.pt`**: Best validation loss (for inference)

Load in Python:
```python
checkpoint = torch.load("checkpoints/best.pt", map_location="cpu")
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
step = checkpoint["step"]
```

## 🛑 Troubleshooting

### Dataset not preparing
```bash
# Monitor progress
./wait_for_dataset.sh

# Check manually
ls -lh data/
```

### Training produces NaN
✓ **Fixed**: Lowered learning rate to 1.5e-4 and increased warmup

### Model outputs gibberish
✓ **Fixed**: Trained on larger dataset (WikiText-103 vs WikiText-2)

### Out of memory
- Reduce `MICRO_BATCH` to 1 (slower but less VRAM)
- Reduce `BLOCK_SIZE` to 64
- Remove gradient checkpointing

### GPU not detected
```python
# Check in Python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # GPU name
```

## 📚 References

- **Mixture of Experts**: [Switch Transformers](https://arxiv.org/abs/2101.03961)
- **GPT Architecture**: [Language Models are Unsupervised Multitask Learners](https://d4mucfpkswtq.cloudfront.net/better-language-models/language-models.pdf)
- **Memory Optimization**: [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)
- **Tokenization**: [tiktoken](https://github.com/openai/tiktoken)

## 📝 License

MIT License - See LICENSE file

---

**Status**: ✅ Ready for training!

Next steps:
1. ⏳ Wait for dataset preparation (`prepare_data.py`)
2. ▶️ Run training (`python main.py`)
3. 🎉 Generate text (`python run.py`)

```
`data/meta.txt`:

```txt
dataset      : fineweb
tokeniser    : gpt2
vocab_size   : 50257
train_tokens : 9753076758
val_tokens : 100388902
test_tokens : 100523684

```
`deepspeed.md`:

```md

Alright, let’s get practical. DeepSpeed is basically a **booster pack for training large models with PyTorch**. It handles memory, distributes work across GPUs, and keeps things from melting your hardware. 🧠⚡

We’ll walk through the **minimal working setup** so the idea clicks.

---

## 1. Install DeepSpeed

First step: install it.

```bash
pip install deepspeed
```

If CUDA and PyTorch are already installed, DeepSpeed compiles its optimizations automatically.

Check it works:

```bash
deepspeed --version
```

---

## 2. Simple PyTorch Training Script

A normal PyTorch training script might look like this:

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for step in range(100):
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)

    output = model(x)
    loss = ((output - y) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

DeepSpeed wraps this training loop so it can **handle distributed training and memory optimization**.

---

## 3. Add DeepSpeed to the Script

Modify the script like this:

```python
import torch
import torch.nn as nn
import deepspeed

model = nn.Linear(10, 1)

parameters = filter(lambda p: p.requires_grad, model.parameters())

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=parameters,
    config="ds_config.json"
)

for step in range(100):
    x = torch.randn(32, 10).to(model_engine.local_rank)
    y = torch.randn(32, 1).to(model_engine.local_rank)

    output = model_engine(x)
    loss = ((output - y) ** 2).mean()

    model_engine.backward(loss)
    model_engine.step()
```

Notice the difference:

Instead of
`loss.backward()`
you use

```
model_engine.backward(loss)
```

DeepSpeed now manages **gradient sync, memory, and distributed GPUs**.

---

## 4. Create the DeepSpeed Config

DeepSpeed uses a JSON config file.

`ds_config.json`

```json
{
  "train_batch_size": 32,
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2
  }
}
```

Key parts:

**fp16**
Uses half precision to save memory.

**zero_optimization**

* stage 1 → optimizer states split
* stage 2 → gradients split
* stage 3 → full model partitioned

Stage 3 is the **big guns for massive models**.

---

## 5. Run the Training

Instead of running Python directly, you launch with DeepSpeed:

```bash
deepspeed train.py
```

Multi-GPU example:

```bash
deepspeed --num_gpus=4 train.py
```

Now your model trains **distributed across GPUs automatically**.

---

## 6. Example with Hugging Face Transformers

DeepSpeed is heavily used with **Hugging Face Transformers**.

Example training command:

```bash
deepspeed run_clm.py \
  --model_name_or_path gpt2 \
  --deepspeed ds_config.json
```

This is how people train **LLMs efficiently**.

---

## What DeepSpeed Is Really Doing Behind the Scenes

When you enable ZeRO optimization, it:

* splits model parameters across GPUs
* shards gradients
* shards optimizer states
* optionally offloads memory to CPU or NVMe

So your GPU doesn’t have to hold **the entire model at once**.

That’s how researchers train **100B+ parameter models** without needing a ridiculous cluster.

---

## A Mental Model

Think of training a huge neural network like moving a giant couch up a staircase.

Without DeepSpeed:

one person tries to carry the couch alone. 💀

With DeepSpeed:

four people lift different corners and move together. 🛋️

Same couch.
Way less suffering.

---

One spicy insight: DeepSpeed becomes **really powerful when models hit GPU memory limits**. That’s why it’s everywhere in LLM training.

The next level trick is combining it with **LoRA fine-tuning**, which lets you train huge models on **a single GPU or even a laptop**. That combo is what a lot of modern AI hackers use. 🔬

```
`ds_config.active.json`:

```json
{
  "train_batch_size": 4,
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 4,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 0.00015,
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-08,
      "weight_decay": 0.01,
      "torch_adam": true
    }
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": false
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 1000000.0,
    "gather_16bit_weights_on_model_save": false
  },
  "bf16": {
    "enabled": true
  },
  "gradient_clipping": 1.0,
  "activation_checkpointing": {
    "partition_activations": true,
    "contiguous_memory_optimization": true,
    "number_checkpoints": 12,
    "synchronize_checkpoint_boundary": false,
    "cpu_checkpointing": true
  },
  "wall_clock_breakdown": false,
  "steps_per_print": 100,
  "fp16": {
    "enabled": false
  },
  "amp": {
    "enabled": false,
    "amp_master_weights": false,
    "loss_scale_window": 1000
  }
}
```
`ds_config.json`:

```json
{
  "train_batch_size": 4,
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 4,
  
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1.5e-4,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01,
      "torch_adam": true
    }
  },
  
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": false
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 1e6,
    "gather_16bit_weights_on_model_save": false
  },
  
  "bf16": {
    "enabled": true
  },
  
  "gradient_clipping": 1.0,
  
  "activation_checkpointing": {
    "partition_activations": true,
    "contiguous_memory_optimization": true,
    "number_checkpoints": 12,
    "synchronize_checkpoint_boundary": false,
    "cpu_checkpointing": true
  },
  
  "wall_clock_breakdown": false,
  
  "steps_per_print": 100,
  
  "fp16": {
    "enabled": false
  },
  
  "amp": {
    "enabled": false,
    "amp_master_weights": false,
    "loss_scale_window": 1000
  }
}

```
`main.py`:

```py
"""
MoE GPT – 0.5 Billion Parameter Language Model
================================================
Mixture-of-Experts GPT trained on WikiText-2.
Fits on GPUs with as little as 4 GB VRAM via:
  - FP16 model weights on GPU  (~1 GB)
  - CPU-offloaded AdamW         (optimizer states on RAM, not VRAM)
  - Gradient checkpointing      (recompute activations to save memory)

Architecture
  12 Transformer layers  ×  (12-head attention  +  MoE FFN)
  8 expert FFNs per layer, top-2 routing
  Total params  ≈ 521 M   |   Active per token  ≈ 180 M

Run order:
    pip install torch tiktoken numpy datasets
    python prepare_data.py          # once — downloads WikiText-2
    python main.py                  # train + generate
"""

import os
import math
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import tiktoken
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn,
    SpinnerColumn, MofNCompleteColumn,
)
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()

# ═════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA  (memory-mapped .bin files from prepare_data.py)
# ═════════════════════════════════════════════════════════════════════════════

DATA_DIR = "data"
for split in ("train", "val", "test"):
    path = os.path.join(DATA_DIR, f"{split}.bin")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ERROR] '{path}' not found.\n"
            "Run  python prepare_data.py  first."
        )

train_data = np.memmap(os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16, mode="r")
val_data   = np.memmap(os.path.join(DATA_DIR, "val.bin"),   dtype=np.uint16, mode="r")
test_data  = np.memmap(os.path.join(DATA_DIR, "test.bin"),  dtype=np.uint16, mode="r")

print("Dataset loaded (memory-mapped)")
print(f"  Train : {len(train_data):>12,} tokens")
print(f"  Val   : {len(val_data):>12,} tokens")
print(f"  Test  : {len(test_data):>12,} tokens")
print()

# ═════════════════════════════════════════════════════════════════════════════
# 2. TOKENISER – GPT-2 BPE  (matches prepare_data.py)
# ═════════════════════════════════════════════════════════════════════════════

enc        = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab                      # 50 257

def encode(text: str) -> list:
    return enc.encode_ordinary(text)

def decode(ids: list) -> str:
    return enc.decode(ids)

print(f"Tokeniser : GPT-2 BPE  (vocab {vocab_size:,})")
print()

# ═════════════════════════════════════════════════════════════════════════════
# 3. HYPERPARAMETERS
# ═════════════════════════════════════════════════════════════════════════════

BLOCK_SIZE    = 512              # context window (tokens)
MICRO_BATCH   = 2                # samples per GPU forward pass (tiny for VRAM)
GRAD_ACCUM    = 8                # accumulate before optimizer step → eff. batch 16
EMBED_DIM     = 768              # model width
NUM_HEADS     = 12               # attention heads
NUM_LAYERS    = 12               # transformer blocks
NUM_EXPERTS   = 8                # expert FFNs per MoE layer
TOP_K         = 2                # experts activated per token
FFN_DIM       = EMBED_DIM * 4   # 3 072  (expert hidden dim)
DROPOUT       = 0.1
LR            = 1.5e-4           # peak learning rate (reduced from 3e-4 to prevent NaN)
WARMUP_STEPS  = 500              # increased warmup for stability
MAX_ITERS     = 100_000           # extended training to 100k steps
EVAL_EVERY    = 2_000
EVAL_ITERS    = 50
AUX_LOSS_W    = 0.01             # load-balancing auxiliary loss weight
GRAD_CLIP     = 1.0
CHECKPOINT_DIR = "checkpoints"   # directory for saving checkpoints

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32
# bfloat16: same exponent range as fp32 — no overflow/NaN, no GradScaler needed.
# float16 caused NaN because it overflows at 65504.

print(f"Device          : {DEVICE.upper()}")
print(f"Precision       : {'BF16 + CPU-offload optimizer' if DTYPE == torch.bfloat16 else 'FP32'}")
print(f"Effective batch : {MICRO_BATCH * GRAD_ACCUM}")
print()

# ═════════════════════════════════════════════════════════════════════════════
# 4. DATA LOADER
# ═════════════════════════════════════════════════════════════════════════════

def get_batch(split="train"):
    data = {"train": train_data, "val": val_data, "test": test_data}[split]
    ix = np.random.randint(0, len(data) - BLOCK_SIZE, size=(MICRO_BATCH,))
    x = np.stack([data[i   : i + BLOCK_SIZE    ].astype(np.int64) for i in ix])
    y = np.stack([data[i+1 : i + BLOCK_SIZE + 1].astype(np.int64) for i in ix])
    return torch.from_numpy(x).to(DEVICE), torch.from_numpy(y).to(DEVICE)

# ═════════════════════════════════════════════════════════════════════════════
# 5. MODEL — Mixture-of-Experts GPT  (~0.5 B params)
# ═════════════════════════════════════════════════════════════════════════════

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with fused QKV projection."""

    def __init__(self):
        super().__init__()
        self.n_heads  = NUM_HEADS
        self.head_dim = EMBED_DIM // NUM_HEADS
        self.qkv      = nn.Linear(EMBED_DIM, 3 * EMBED_DIM, bias=False)
        self.proj      = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.attn_drop = nn.Dropout(DROPOUT)
        self.proj_drop = nn.Dropout(DROPOUT)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
                .view(1, 1, BLOCK_SIZE, BLOCK_SIZE),
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)          # each (B, H, T, D)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att.float(), dim=-1).to(x.dtype)   # softmax in fp32
        att = self.attn_drop(att)

        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj_drop(self.proj(out))


class ExpertFFN(nn.Module):
    """Single expert: two-layer FFN with GELU."""

    def __init__(self):
        super().__init__()
        self.w1   = nn.Linear(EMBED_DIM, FFN_DIM)
        self.w2   = nn.Linear(FFN_DIM, EMBED_DIM)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        return self.drop(self.w2(self.act(self.w1(x))))


class MoELayer(nn.Module):
    """
    Mixture-of-Experts: routes each token to TOP_K of NUM_EXPERTS FFNs.
    Includes Switch-Transformer-style load-balancing auxiliary loss.
    """

    def __init__(self):
        super().__init__()
        self.router  = nn.Linear(EMBED_DIM, NUM_EXPERTS, bias=False)
        self.experts = nn.ModuleList([ExpertFFN() for _ in range(NUM_EXPERTS)])

    def forward(self, x):
        B, T, C = x.shape
        flat = x.reshape(-1, C)                              # (N, C)
        N = flat.shape[0]

        # ── routing ──
        logits = self.router(flat)                            # (N, E)
        probs  = F.softmax(logits.float(), dim=-1)            # fp32 for stability

        top_w, top_i = torch.topk(probs, TOP_K, dim=-1)      # (N, K)
        top_w = (top_w / top_w.sum(dim=-1, keepdim=True)).to(x.dtype)

        # ── load-balancing loss ──
        one_hot = F.one_hot(top_i, NUM_EXPERTS).float().sum(dim=1)   # (N, E)
        f = one_hot.mean(dim=0)
        P = probs.mean(dim=0)
        aux_loss = NUM_EXPERTS * (f * P).sum()

        # ── dispatch to experts ──
        out = torch.zeros_like(flat)
        for i, expert in enumerate(self.experts):
            mask = (top_i == i).any(dim=-1)                   # (N,)
            if not mask.any():
                continue
            tokens  = flat[mask]                              # (n_i, C)
            e_out   = expert(tokens)                          # (n_i, C)
            match   = (top_i[mask] == i).to(x.dtype)          # (n_i, K)
            weights = (top_w[mask] * match).sum(-1, keepdim=True)
            out[mask] += weights * e_out

        return out.reshape(B, T, C), aux_loss


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: Attention + MoE, with residuals."""

    def __init__(self):
        super().__init__()
        self.ln1  = nn.LayerNorm(EMBED_DIM)
        self.attn = CausalSelfAttention()
        self.ln2  = nn.LayerNorm(EMBED_DIM)
        self.moe  = MoELayer()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        moe_out, aux = self.moe(self.ln2(x))
        x = x + moe_out
        return x, aux


class MoEGPT(nn.Module):
    """
    Full MoE-GPT model (~521 M parameters, ~180 M active per token).

    1. Token + positional embeddings
    2. 12 × Transformer blocks  (self-attention + MoE FFN)
    3. Final layer-norm → linear head (weight-tied with token embedding)
    """

    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, EMBED_DIM)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        self.drop    = nn.Dropout(DROPOUT)
        self.blocks  = nn.ModuleList([TransformerBlock() for _ in range(NUM_LAYERS)])
        self.ln_f    = nn.LayerNorm(EMBED_DIM)
        self.head    = nn.Linear(EMBED_DIM, vocab_size, bias=False)

        # Weight tying saves ~38 M params and improves training
        self.head.weight = self.tok_emb.weight
        self._init_weights()

    def _init_weights(self):
        """GPT-2-style init with scaled residual projections."""
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(p)
        scale = (2 * NUM_LAYERS) ** -0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=0.02 * scale)
            for expert in block.moe.experts:
                nn.init.normal_(expert.w2.weight, mean=0.0, std=0.02 * scale)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(
            self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        )

        total_aux = 0.0
        for block in self.blocks:
            if self.training:
                x, aux = grad_checkpoint(block, x, use_reentrant=False)
            else:
                x, aux = block(x)
            total_aux = total_aux + aux

        logits = self.head(self.ln_f(x))

        loss = None
        if targets is not None:
            ce   = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
            loss = ce + AUX_LOSS_W * total_aux
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens=200, temperature=0.8, top_k=50, top_p=0.9):
        self.eval()
        ids = encode(prompt)
        idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)

        for _ in range(max_new_tokens):
            ctx = idx[:, -BLOCK_SIZE:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :].float() / temperature
            
            # Top-K filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")
            
            # Top-P (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumsum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumsum_probs > top_p
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = float("-inf")
            
            probs  = F.softmax(logits, dim=-1)
            nxt    = torch.multinomial(probs, 1)
            idx    = torch.cat([idx, nxt], dim=1)

        self.train()
        return decode(idx[0].tolist())

# ═════════════════════════════════════════════════════════════════════════════
# 6. CPU-OFFLOAD OPTIMIZER
#    Hand-rolled AdamW with fp32 master weights but fp16 momentum/variance.
#    Saves ~2 GB CPU RAM compared to using torch.optim.AdamW (all fp32).
#    GPU VRAM cost ≈ 1 GB  (only fp16 model weights + grads).
#
#    Memory breakdown for 520 M params:
#      fp32 master weights : ~2.0 GB
#      fp16 momentum       : ~1.0 GB
#      fp16 variance       : ~1.0 GB
#      Total CPU RAM       : ~4.1 GB  (was ~6.2 GB with all-fp32 AdamW)
# ═════════════════════════════════════════════════════════════════════════════

class CPUOffloadAdamW:
    """
    AdamW with ALL state (master weights, momentum, variance) in fp32 on CPU.
    fp16 m/v was the culprit for NaN — Adam variance accumulates squared
    gradients that easily exceed fp16 max (65504) → overflow → NaN.
    GPU holds only fp16 model weights + fp16 gradients (~1 GB VRAM).
    CPU RAM: fp32 master(2 GB) + fp32 m(2 GB) + fp32 v(2 GB) ≈ 6.2 GB.
    Expose param_groups so torch.amp.GradScaler.unscale_() works correctly.
    """

    def __init__(self, gpu_params, lr=3e-4, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.01):
        self.gpu_params = list(gpu_params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0

        # fp32 master copies + fp32 momentum/variance on CPU
        self.master = [p.data.float().cpu() for p in self.gpu_params]
        self.m = [torch.zeros_like(mp) for mp in self.master]   # fp32
        self.v = [torch.zeros_like(mp) for mp in self.master]   # fp32

        # GradScaler compatibility: unscale_() iterates param_groups
        self.param_groups = [{"params": self.gpu_params}]

    def step(self):
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t

        for i, gp in enumerate(self.gpu_params):
            if gp.grad is None:
                continue
            g = gp.grad.data.float().cpu()   # fp16 grad → fp32

            # Decoupled weight decay
            self.master[i].mul_(1.0 - self.lr * self.wd)

            # Adam moments (all fp32 — no overflow risk)
            self.m[i].mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
            self.v[i].mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)

            # Bias-corrected parameter update
            self.master[i].addcdiv_(
                self.m[i] / bc1,
                (self.v[i] / bc2).sqrt_().add_(self.eps),
                value=-self.lr,
            )

            # Push updated fp32 weights → GPU fp16
            gp.data.copy_(self.master[i])

    def zero_grad(self):
        for gp in self.gpu_params:
            gp.grad = None

    def set_lr(self, lr):
        self.lr = lr

    def state_dict(self):
        return {"t": self.t, "master": self.master, "m": self.m, "v": self.v}

    def load_state_dict(self, sd):
        self.t = sd["t"]
        self.master = sd["master"]
        self.m = sd["m"]
        self.v = sd["v"]
        for gp, mp in zip(self.gpu_params, self.master):
            gp.data.copy_(mp.data)

# ═════════════════════════════════════════════════════════════════════════════
# 7. CHECKPOINT HELPERS
# ═════════════════════════════════════════════════════════════════════════════

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(step, model, optimizer, train_loss, val_loss, path):
    """Save model + optimizer + training state to disk."""
    torch.save({
        "step":        step,
        "model":       model.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "train_loss":  train_loss,
        "val_loss":    val_loss,
    }, path)

def load_checkpoint(path, model, optimizer):
    """Load checkpoint and return the step to resume from."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"  Resumed from step {ckpt['step']}  "
          f"(train {ckpt['train_loss']:.4f}, val {ckpt['val_loss']:.4f})")
    return ckpt["step"], ckpt["val_loss"]

# ═════════════════════════════════════════════════════════════════════════════
# 8. LEARNING-RATE SCHEDULE  (linear warmup → cosine decay to 10 %)
# ═════════════════════════════════════════════════════════════════════════════

def get_lr(step):
    if step < WARMUP_STEPS:
        return LR * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, MAX_ITERS - WARMUP_STEPS)
    return LR * 0.1 + 0.5 * LR * 0.9 * (1 + math.cos(math.pi * progress))

# ═════════════════════════════════════════════════════════════════════════════
# 9. LOSS ESTIMATION
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(EVAL_ITERS):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

# ═════════════════════════════════════════════════════════════════════════════
# 10. INSTANTIATE MODEL + OPTIMIZER
# ═════════════════════════════════════════════════════════════════════════════

if DEVICE == "cuda":
    torch.cuda.empty_cache()

# ── Delete any NaN-poisoned checkpoints before loading ──
_nan_guard = os.path.join(CHECKPOINT_DIR, "latest.pt")
if os.path.exists(_nan_guard):
    try:
        _c = torch.load(_nan_guard, map_location="cpu", weights_only=False)
        if _c.get("val_loss") != _c.get("val_loss"):  # nan != nan
            os.remove(_nan_guard)
            _best = os.path.join(CHECKPOINT_DIR, "best.pt")
            if os.path.exists(_best):
                os.remove(_best)
            print("[yellow]NaN checkpoint detected and removed — starting fresh.[/yellow]")
    except Exception:
        pass

model = MoEGPT()
n_total  = sum(p.numel() for p in model.parameters())
_expert1 = sum(p.numel() for p in model.blocks[0].moe.experts[0].parameters())
n_active = n_total - _expert1 * (NUM_EXPERTS - TOP_K) * NUM_LAYERS

# Move to GPU in fp16  (or stay fp32 on CPU)
model = model.to(dtype=DTYPE, device=DEVICE)
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()
    vram_used = torch.cuda.memory_allocated() / 1024**3
    print(f"GPU VRAM used   : {vram_used:.2f} GiB  (model weights)")

if DEVICE == "cuda":
    # Initialize optimizer AFTER config changes so it uses the new LR
    optimizer = CPUOffloadAdamW(model.parameters(), lr=LR)
    gc.collect()
    opt_gb = n_total * 4 * 3 / 1024**3   # fp32 master + fp32 m + fp32 v
    print(f"CPU RAM for opt : ~{opt_gb:.1f} GiB  (fp32 master + fp32 m + fp32 v)")
else:
    _inner = torch.optim.AdamW(model.parameters(), lr=LR)
    class _Wrap:
        def __init__(self, o): self.opt = o
        def step(self):       self.opt.step()
        def zero_grad(self):  self.opt.zero_grad(set_to_none=True)
        def set_lr(self, lr):
            for pg in self.opt.param_groups: pg["lr"] = lr
        def state_dict(self):       return self.opt.state_dict()
        def load_state_dict(self, sd): self.opt.load_state_dict(sd)
    optimizer = _Wrap(_inner)

print(f"Total  parameters : {n_total:>14,}")
print(f"Active per token  : {n_active:>14,}")
print()

# ── Auto-resume from latest checkpoint ──
RESUME = True  # Automatically resume from latest.pt if it exists
start_step = 0
best_val   = float("inf")
latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest.pt")
if RESUME and os.path.exists(latest_ckpt):
    try:
        _c = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        # Skip NaN-poisoned checkpoints
        if _c.get("val_loss") != _c.get("val_loss") or _c.get("train_loss") != _c.get("train_loss"):
            print("Checkpoint has NaN losses — deleting and starting fresh")
            os.remove(latest_ckpt)
        else:
            print("Checkpoint found — resuming …")
            start_step, best_val = load_checkpoint(latest_ckpt, model, optimizer)
            print()
    except Exception as e:
        print(f"Checkpoint corrupted ({e}) — starting fresh")
else:
    if RESUME:
        print("No checkpoint found — starting fresh training")
    print()

# ═════════════════════════════════════════════════════════════════════════════
# 11. TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════

console.rule("[bold green]Training started")
print()

with Progress(
    SpinnerColumn(),
    TextColumn("[bold blue]{task.description}"),
    BarColumn(bar_width=30),
    MofNCompleteColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
    TextColumn("•"),
    TimeRemainingColumn(),
    TextColumn("•"),
    TextColumn("[yellow]loss {task.fields[train_loss]}"),
    TextColumn("[cyan]val {task.fields[val_loss]}"),
    TextColumn("[magenta]lr {task.fields[lr]}"),
    console=console,
    refresh_per_second=4,
) as progress:
    total_steps = MAX_ITERS - start_step
    task = progress.add_task(
        "Training", total=total_steps,
        train_loss="--.----", val_loss="--.----", lr="--.------",
    )

    for step in range(start_step + 1, MAX_ITERS + 1):

        lr = get_lr(step)
        optimizer.set_lr(lr)

        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(GRAD_ACCUM):
            x, y = get_batch("train")
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=(DTYPE == torch.bfloat16)):
                _, loss = model(x, y)
            (loss / GRAD_ACCUM).backward()
            accum_loss += loss.item() / GRAD_ACCUM

        # Gradient clipping with stricter threshold to prevent explosion
        norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        if norm_before > GRAD_CLIP:
            progress.console.print(f"  [yellow]Gradient norm clipped: {norm_before:.2f} → {GRAD_CLIP}[/]", style="dim")
        optimizer.step()

        progress.update(
            task, advance=1,
            train_loss=f"{accum_loss:.4f}", lr=f"{lr:.6f}",
        )

        if step % EVAL_EVERY == 0 or step == 1:
            losses = estimate_loss()
            progress.update(
                task,
                train_loss=f"{losses['train']:.4f}",
                val_loss=f"{losses['val']:.4f}",
                lr=f"{lr:.6f}",
            )
            progress.console.print(
                f"  [bold]Step {step:>5}[/]  │  "
                f"[yellow]Train {losses['train']:.4f}[/]  │  "
                f"[cyan]Val {losses['val']:.4f}[/]  │  "
                f"[magenta]LR {lr:.6f}[/]"
            )

            # ── Save checkpoints ──
            save_checkpoint(
                step, model, optimizer,
                losses["train"], losses["val"],
                os.path.join(CHECKPOINT_DIR, "latest.pt"),
            )
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(
                    step, model, optimizer,
                    losses["train"], losses["val"],
                    os.path.join(CHECKPOINT_DIR, "best.pt"),
                )
                progress.console.print(
                    f"  [bold green]★ New best val loss: {best_val:.4f}  (saved best.pt)[/]"
                )

print()
console.rule("[bold green]Training complete")
print()

# ── Load best checkpoint for final evaluation ──
best_ckpt = os.path.join(CHECKPOINT_DIR, "best.pt")
if os.path.exists(best_ckpt):
    print("Loading best checkpoint for evaluation …")
    load_checkpoint(best_ckpt, model, optimizer)
    print()

# ═════════════════════════════════════════════════════════════════════════════
# 12. TEST EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

model.eval()
test_losses = []
with torch.no_grad():
    for _ in range(EVAL_ITERS):
        x, y = get_batch("test")
        _, loss = model(x, y)
        test_losses.append(loss.item())
test_loss = sum(test_losses) / len(test_losses)
print(f"Test loss : {test_loss:.4f}")
print()
model.train()

# ═════════════════════════════════════════════════════════════════════════════
# 13. TEXT GENERATION SAMPLES
# ═════════════════════════════════════════════════════════════════════════════

prompts = [
    "The history of",
    "Scientists have discovered",
    "In the early twentieth century",
]

print("=" * 60)
print("Generated Text Samples")
print("=" * 60)

for prompt in prompts:
    output = model.generate(prompt, max_new_tokens=120, temperature=0.7)
    print(f"\nPrompt : \"{prompt}\"")
    print(f"Output : {output.strip()}")
    print()

# ═════════════════════════════════════════════════════════════════════════════
# 14. INTERACTIVE MODE
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Interactive Mode  (type 'quit' to exit)")
print("=" * 60)

while True:
    try:
        prompt = input("\nEnter a prompt: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not prompt or prompt.lower() == "quit":
        break
    output = model.generate(prompt, max_new_tokens=150, temperature=0.8)
    print(f"\n{output.strip()}")

print("\nGoodbye!")

```
`main_deepspeed.py`:

```py
"""
MoE GPT – 0.5 Billion Parameter Language Model (DeepSpeed ZeRO-3)
==================================================================
Mixture-of-Experts GPT trained on FineWeb-Edu with DeepSpeed ZeRO-Infinity:
  - ZeRO Stage 3: All states partitioned across GPUs + CPU RAM offload
  - CPU Offloading: Parameters & optimizer states in CPU RAM
  - Memory efficient: Fits massive models on limited VRAM
  - Automatic gradient checkpointing & mixed precision (bfloat16)

Architecture
  12 Transformer layers  ×  (12-head attention  +  MoE FFN)
  8 expert FFNs per layer, top-2 routing
  Total params  ≈ 521 M   |   Active per token  ≈ 180 M

Run order:
    pip install torch tiktoken numpy datasets deepspeed
    python prepare_data.py          # once — downloads FineWeb-Edu
    deepspeed --num_gpus 1 main.py  # train with DeepSpeed
    python run.py                   # generate
"""

import os
import sys
import math
import gc
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import tiktoken
import deepspeed
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn,
    SpinnerColumn, MofNCompleteColumn,
)
from rich.console import Console

console = Console()
IST = ZoneInfo("Asia/Kolkata")

# ═════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA  (memory-mapped .bin files from prepare_data.py)
# ═════════════════════════════════════════════════════════════════════════════

DATA_DIR = "data"
for split in ("train", "val", "test"):
    path = os.path.join(DATA_DIR, f"{split}.bin")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ERROR] '{path}' not found.\n"
            "Run  python prepare_data.py  first."
        )

train_data = np.memmap(os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16, mode="r")
val_data   = np.memmap(os.path.join(DATA_DIR, "val.bin"),   dtype=np.uint16, mode="r")
test_data  = np.memmap(os.path.join(DATA_DIR, "test.bin"),  dtype=np.uint16, mode="r")

print("Dataset loaded (memory-mapped)")
print(f"  Train : {len(train_data):>12,} tokens")
print(f"  Val   : {len(val_data):>12,} tokens")
print(f"  Test  : {len(test_data):>12,} tokens")
print()

# ═════════════════════════════════════════════════════════════════════════════
# 2. TOKENISER – GPT-2 BPE  (matches prepare_data.py)
# ═════════════════════════════════════════════════════════════════════════════

enc        = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab                      # 50 257

def encode(text: str) -> list:
    return enc.encode_ordinary(text)

def decode(ids: list) -> str:
    return enc.decode(ids)

print(f"Tokeniser : GPT-2 BPE  (vocab {vocab_size:,})")
print()

# ═════════════════════════════════════════════════════════════════════════════
# 3. HYPERPARAMETERS
# ═════════════════════════════════════════════════════════════════════════════

BLOCK_SIZE    = 64               # context window (tokens)
MICRO_BATCH   = 8                # samples per GPU forward pass (managed by DeepSpeed)
GRAD_ACCUM    = 4                # accumulate before optimizer step → eff. batch 16
EMBED_DIM     = 512              # model width
NUM_HEADS     = 8                # attention heads
NUM_LAYERS    = 8                # transformer blocks
NUM_EXPERTS   = 4                # expert FFNs per MoE layer
TOP_K         = 2                # experts activated per token
FFN_DIM       = EMBED_DIM * 4   # 2 048  (expert hidden dim)
DROPOUT       = 0.1
LR            = 1.5e-4           # peak learning rate
WARMUP_STEPS  = 500              # linear warmup
MAX_ITERS     = 120_000           # total optimiser steps
EVAL_EVERY    = 2_000
EVAL_ITERS    = 50
AUX_LOSS_W    = 0.01             # load-balancing auxiliary loss weight
GRAD_CLIP     = 1.0
CHECKPOINT_DIR = "checkpoints"   # directory for saving checkpoints

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32

ALLOW_TF32 = os.environ.get("ALLOW_TF32", "1") == "1"
USE_TORCH_COMPILE = os.environ.get("USE_TORCH_COMPILE", "0") == "1"
USE_ACTIVATION_CHECKPOINT = os.environ.get("USE_ACTIVATION_CHECKPOINT", "1") == "1"

if DEVICE == "cuda":
    # Throughput-oriented CUDA settings.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32
    torch.backends.cudnn.allow_tf32 = ALLOW_TF32
    torch.set_float32_matmul_precision("high")

print(f"Device          : {DEVICE.upper()}")
print(f"Precision       : {'BF16 (DeepSpeed)' if DTYPE == torch.bfloat16 else 'FP32'}")
print(f"Effective batch (default) : {MICRO_BATCH * GRAD_ACCUM}")
print(f"TF32            : {'ON' if (DEVICE == 'cuda' and ALLOW_TF32) else 'OFF'}")
print(f"Torch compile   : {'ON' if USE_TORCH_COMPILE else 'OFF'}")
print(f"Act checkpoint  : {'ON' if USE_ACTIVATION_CHECKPOINT else 'OFF'}")
print()

# ═════════════════════════════════════════════════════════════════════════════
# 4. DATA LOADER
# ═════════════════════════════════════════════════════════════════════════════

def get_batch(split="train"):
    data = {"train": train_data, "val": val_data, "test": test_data}[split]
    ix = np.random.randint(0, len(data) - BLOCK_SIZE, size=(MICRO_BATCH,))
    x = np.stack([data[i   : i + BLOCK_SIZE    ].astype(np.int64) for i in ix])
    y = np.stack([data[i+1 : i + BLOCK_SIZE + 1].astype(np.int64) for i in ix])
    return torch.from_numpy(x).to(DEVICE), torch.from_numpy(y).to(DEVICE)

# ═════════════════════════════════════════════════════════════════════════════
# 5. MODEL — Mixture-of-Experts GPT  (~0.5 B params)
# ═════════════════════════════════════════════════════════════════════════════

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with fused QKV projection."""

    def __init__(self):
        super().__init__()
        self.n_heads  = NUM_HEADS
        self.head_dim = EMBED_DIM // NUM_HEADS
        self.qkv      = nn.Linear(EMBED_DIM, 3 * EMBED_DIM, bias=False)
        self.proj      = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.attn_drop = nn.Dropout(DROPOUT)
        self.proj_drop = nn.Dropout(DROPOUT)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
                .view(1, 1, BLOCK_SIZE, BLOCK_SIZE),
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)          # each (B, H, T, D)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att.float(), dim=-1).to(x.dtype)   # softmax in fp32
        att = self.attn_drop(att)

        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj_drop(self.proj(out))


class ExpertFFN(nn.Module):
    """Single expert: two-layer FFN with GELU."""

    def __init__(self):
        super().__init__()
        self.w1   = nn.Linear(EMBED_DIM, FFN_DIM)
        self.w2   = nn.Linear(FFN_DIM, EMBED_DIM)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        return self.drop(self.w2(self.act(self.w1(x))))


class MoELayer(nn.Module):
    """
    Mixture-of-Experts: routes each token to TOP_K of NUM_EXPERTS FFNs.
    Includes Switch-Transformer-style load-balancing auxiliary loss.
    """

    def __init__(self):
        super().__init__()
        self.router  = nn.Linear(EMBED_DIM, NUM_EXPERTS, bias=False)
        self.experts = nn.ModuleList([ExpertFFN() for _ in range(NUM_EXPERTS)])

    def forward(self, x):
        B, T, C = x.shape
        flat = x.reshape(-1, C)                              # (N, C)
        N = flat.shape[0]

        # ── routing ──
        logits = self.router(flat)                            # (N, E)
        probs  = F.softmax(logits.float(), dim=-1)            # fp32 for stability

        top_w, top_i = torch.topk(probs, TOP_K, dim=-1)      # (N, K)
        top_w = (top_w / top_w.sum(dim=-1, keepdim=True)).to(x.dtype)

        # ── load-balancing loss ──
        one_hot = F.one_hot(top_i, NUM_EXPERTS).float().sum(dim=1)   # (N, E)
        f = one_hot.mean(dim=0)
        P = probs.mean(dim=0)
        aux_loss = NUM_EXPERTS * (f * P).sum()

        # ── dispatch to experts ──
        out = torch.zeros_like(flat)
        for i, expert in enumerate(self.experts):
            mask = (top_i == i).any(dim=-1)                   # (N,)
            if not mask.any():
                continue
            tokens  = flat[mask]                              # (n_i, C)
            e_out   = expert(tokens)                          # (n_i, C)
            match   = (top_i[mask] == i).to(x.dtype)          # (n_i, K)
            weights = (top_w[mask] * match).sum(-1, keepdim=True)
            out[mask] += weights * e_out

        return out.reshape(B, T, C), aux_loss


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: Attention + MoE, with residuals."""

    def __init__(self):
        super().__init__()
        self.ln1  = nn.LayerNorm(EMBED_DIM)
        self.attn = CausalSelfAttention()
        self.ln2  = nn.LayerNorm(EMBED_DIM)
        self.moe  = MoELayer()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        moe_out, aux = self.moe(self.ln2(x))
        x = x + moe_out
        return x, aux


class MoEGPT(nn.Module):
    """
    Full MoE-GPT model (~521 M parameters, ~180 M active per token).

    1. Token + positional embeddings
    2. 12 × Transformer blocks  (self-attention + MoE FFN)
    3. Final layer-norm → linear head (weight-tied with token embedding)
    """

    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, EMBED_DIM)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        self.drop    = nn.Dropout(DROPOUT)
        self.blocks  = nn.ModuleList([TransformerBlock() for _ in range(NUM_LAYERS)])
        self.ln_f    = nn.LayerNorm(EMBED_DIM)
        self.head    = nn.Linear(EMBED_DIM, vocab_size, bias=False)

        # Weight tying saves ~38 M params and improves training
        self.head.weight = self.tok_emb.weight
        self._init_weights()

    def _init_weights(self):
        """GPT-2-style init with scaled residual projections."""
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(p)
        scale = (2 * NUM_LAYERS) ** -0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=0.02 * scale)
            for expert in block.moe.experts:
                nn.init.normal_(expert.w2.weight, mean=0.0, std=0.02 * scale)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(
            self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        )

        total_aux = 0.0
        for block in self.blocks:
            if self.training and USE_ACTIVATION_CHECKPOINT:
                x, aux = grad_checkpoint(block, x, use_reentrant=False)
            else:
                x, aux = block(x)
            total_aux = total_aux + aux

        logits = self.head(self.ln_f(x))

        loss = None
        if targets is not None:
            ce   = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
            loss = ce + AUX_LOSS_W * total_aux
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens=200, temperature=0.8, top_k=50, top_p=0.9):
        self.eval()
        ids = encode(prompt)
        idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)

        for _ in range(max_new_tokens):
            ctx = idx[:, -BLOCK_SIZE:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :].float() / temperature

            # Top-K filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Top-P (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumsum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumsum_probs > top_p
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = float("-inf")

            probs  = F.softmax(logits, dim=-1)
            nxt    = torch.multinomial(probs, 1)
            idx    = torch.cat([idx, nxt], dim=1)

        self.train()
        return decode(idx[0].tolist())

# ═════════════════════════════════════════════════════════════════════════════
# 6. CHECKPOINT HELPERS
# ═════════════════════════════════════════════════════════════════════════════

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def _strip_orig_mod_prefix(state_dict):
    out = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            out[k[len("_orig_mod."):]] = v
        else:
            out[k] = v
    return out


def _add_orig_mod_prefix(state_dict):
    out = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            out[k] = v
        else:
            out[f"_orig_mod.{k}"] = v
    return out


def _align_state_dict_for_model(state_dict, model):
    """Align checkpoint keys with model keys (compiled vs non-compiled)."""
    model_keys = list(model.state_dict().keys())
    if not model_keys:
        return state_dict

    model_has_orig = model_keys[0].startswith("_orig_mod.")
    ckpt_keys = list(state_dict.keys())
    ckpt_has_orig = bool(ckpt_keys) and ckpt_keys[0].startswith("_orig_mod.")

    if model_has_orig and not ckpt_has_orig:
        return _add_orig_mod_prefix(state_dict)
    if not model_has_orig and ckpt_has_orig:
        return _strip_orig_mod_prefix(state_dict)
    return state_dict

def save_checkpoint(step, model, train_loss, val_loss, path):
    """Save model and training state to disk."""
    # DeepSpeed handles checkpointing, but we also save basic metadata
    model_state = model.state_dict() if hasattr(model, "state_dict") else None
    if model_state is not None:
        # Store canonical keys so checkpoints are reusable across compile modes.
        model_state = _strip_orig_mod_prefix(model_state)

    checkpoint = {
        "step": step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "model_state": model_state,
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, model):
    """Load checkpoint and return the step to resume from."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("model_state"):
        model_state = _align_state_dict_for_model(ckpt["model_state"], model)
        model.load_state_dict(model_state)
    print(f"  Resumed from step {ckpt['step']}  "
          f"(train {ckpt['train_loss']:.4f}, val {ckpt['val_loss']:.4f})")
    return ckpt["step"], ckpt["val_loss"]

# ═════════════════════════════════════════════════════════════════════════════
# 7. LEARNING-RATE SCHEDULE  (linear warmup → cosine decay to 10 %)
# ═════════════════════════════════════════════════════════════════════════════

def get_lr(step):
    if step < WARMUP_STEPS:
        return LR * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, MAX_ITERS - WARMUP_STEPS)
    return LR * 0.1 + 0.5 * LR * 0.9 * (1 + math.cos(math.pi * progress))


def get_eta_clock(progress, task_id):
    """Return estimated finish time in IST as HH:MM."""
    remaining = progress.tasks[task_id].time_remaining
    if remaining is None:
        return "--:--"
    end_at = datetime.now(IST) + timedelta(seconds=max(0.0, remaining))
    return end_at.strftime("%H:%M")

# ═════════════════════════════════════════════════════════════════════════════
# 8. LOSS ESTIMATION
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(EVAL_ITERS):
            x, y = get_batch(split)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda" and DTYPE == torch.bfloat16)):
                _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

# ═════════════════════════════════════════════════════════════════════════════
# 9. DEEPSPEED INITIALIZATION & TRAINING
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Clear cache
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Initialize model
    model = MoEGPT()
    if DEVICE == "cuda" and USE_TORCH_COMPILE:
        # Full-graph mode is too brittle for dynamic shapes; max-autotune is a good speed/compat compromise.
        model = torch.compile(model, mode="max-autotune", fullgraph=False)

    n_total  = sum(p.numel() for p in model.parameters())
    _expert1 = sum(p.numel() for p in model.blocks[0].moe.experts[0].parameters())
    n_active = n_total - _expert1 * (NUM_EXPERTS - TOP_K) * NUM_LAYERS

    print(f"Total  parameters : {n_total:>14,}")
    print(f"Active per token  : {n_active:>14,}")
    print()

    # Ensure LOCAL_RANK is set so DeepSpeed's sanity checks pass when running
    # this script directly with `python main_deepspeed.py` (single-GPU).
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"

    # Initialize torch.distributed if not already initialized (single-process setup).
    if not torch.distributed.is_initialized():
        init_kwargs = {
            "backend": "nccl" if DEVICE == "cuda" else "gloo",
            "init_method": "tcp://127.0.0.1:29500",
            "rank": 0,
            "world_size": 1,
        }
        if DEVICE == "cuda":
            init_kwargs["device_id"] = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.distributed.init_process_group(**init_kwargs)

    # Load DeepSpeed config (launcher may provide a mode-specific config path).
    ds_config_path = os.environ.get("DS_CONFIG_PATH", "ds_config.json")
    with open(ds_config_path) as f:
        ds_config = json.load(f)

    # Keep runtime batch settings aligned with DeepSpeed config.
    MICRO_BATCH = int(ds_config.get("train_micro_batch_size_per_gpu", MICRO_BATCH))
    GRAD_ACCUM = int(ds_config.get("gradient_accumulation_steps", GRAD_ACCUM))

    print(f"DeepSpeed micro-batch : {MICRO_BATCH}")
    print(f"DeepSpeed grad accum  : {GRAD_ACCUM}")
    print(f"DeepSpeed eff batch   : {MICRO_BATCH * GRAD_ACCUM}")
    print()

    # Initialize DeepSpeed engine
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=type("args", (), {"local_rank": int(os.environ.get("LOCAL_RANK", 0))})(),
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
        dist_init_required=False,
    )

    print(f"[DeepSpeed] Initialized with ZeRO Stage {ds_config['zero_optimization']['stage']}")
    print(f"[DeepSpeed] Device: {model_engine.device}")
    print()

    # Training state
    start_step = 0
    best_val = float("inf")
    prev_val = None

    # Auto-resume from latest checkpoint (with NaN guard)
    latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest.pt")
    if os.path.exists(latest_ckpt):
        try:
            _c = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
            if _c.get("val_loss") != _c.get("val_loss") or _c.get("train_loss") != _c.get("train_loss"):
                print("Checkpoint has NaN losses — deleting and starting fresh")
                os.remove(latest_ckpt)
            else:
                print("Checkpoint found — resuming …")
                start_step, best_val = load_checkpoint(latest_ckpt, model)
                print()
        except Exception as e:
            print(f"Checkpoint corrupted ({e}) — starting fresh")
    else:
        print("No checkpoint found — starting fresh training")
        print()

    # ─────────────────────────────────────────────────────────────────────────
    # TRAINING LOOP
    # ─────────────────────────────────────────────────────────────────────────

    console.rule("[bold green]Training started (DeepSpeed)")
    print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        TextColumn("•"),
        TextColumn("[green]ETA {task.fields[end_clock]} IST"),
        TextColumn("•"),
        TextColumn("[yellow]loss {task.fields[train_loss]}"),
        TextColumn("[cyan]val {task.fields[val_loss]}"),
        TextColumn("[magenta]lr {task.fields[lr]}"),
        console=console,
        refresh_per_second=4,
    ) as progress:
        total_steps = MAX_ITERS - start_step
        task = progress.add_task(
            "Training", total=total_steps,
            train_loss="--.----", val_loss="--.----", lr="--.------", end_clock="--:--",
        )

        step = start_step
        micro_loss_sum = 0.0
        micro_loss_count = 0
        total_micro_steps = MAX_ITERS * GRAD_ACCUM

        for micro_step in range(start_step * GRAD_ACCUM + 1, total_micro_steps + 1):

            lr = get_lr(step + 1)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            x, y = get_batch("train")
            _, loss = model_engine(x, y)

            model_engine.backward(loss)
            is_boundary = model_engine.is_gradient_accumulation_boundary()
            model_engine.step()

            micro_loss_sum += loss.item()
            micro_loss_count += 1

            if not is_boundary:
                continue

            step += 1
            accum_loss = micro_loss_sum / max(1, micro_loss_count)
            micro_loss_sum = 0.0
            micro_loss_count = 0

            progress.update(
                task,
                advance=1,
                train_loss=f"{accum_loss:.4f}",
                lr=f"{lr:.6f}",
                end_clock=get_eta_clock(progress, task),
            )

            if step % EVAL_EVERY == 0:
                losses = estimate_loss(model_engine.module if hasattr(model_engine, "module") else model_engine)
                if prev_val is None:
                    trend = "init"
                    delta = 0.0
                else:
                    delta = losses["val"] - prev_val
                    if delta < -1e-6:
                        trend = "improving"
                    elif delta > 1e-6:
                        trend = "worse"
                    else:
                        trend = "flat"
                prev_val = losses["val"]

                progress.update(
                    task,
                    train_loss=f"{losses['train']:.4f}",
                    val_loss=f"{losses['val']:.4f}",
                    lr=f"{lr:.6f}",
                    end_clock=get_eta_clock(progress, task),
                )
                progress.console.print(
                    f"  [bold]Step {step:>5}[/]  │  "
                    f"[yellow]Train {losses['train']:.4f}[/]  │  "
                    f"[cyan]Val {losses['val']:.4f} ({trend}, Δ {delta:+.4f})[/]  │  "
                    f"[magenta]LR {lr:.6f}[/]"
                )

                # Save checkpoints
                save_checkpoint(
                    step, model_engine.module if hasattr(model_engine, 'module') else model_engine,
                    losses["train"], losses["val"],
                    os.path.join(CHECKPOINT_DIR, "latest.pt"),
                )
                if losses["val"] < best_val:
                    best_val = losses["val"]
                    save_checkpoint(
                        step, model_engine.module if hasattr(model_engine, 'module') else model_engine,
                        losses["train"], losses["val"],
                        os.path.join(CHECKPOINT_DIR, "best.pt"),
                    )
                    progress.console.print(
                        f"  [bold green]★ New best val loss: {best_val:.4f}  (saved best.pt)[/]"
                    )

            if step >= MAX_ITERS:
                break

    print()
    console.rule("[bold green]Training complete")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # TEST EVALUATION
    # ─────────────────────────────────────────────────────────────────────────

    model_eval = model_engine.module if hasattr(model_engine, 'module') else model_engine
    model_eval.eval()
    test_losses = []
    with torch.no_grad():
        for _ in range(EVAL_ITERS):
            x, y = get_batch("test")
            _, loss = model_eval(x, y)
            test_losses.append(loss.item())
    test_loss = sum(test_losses) / len(test_losses)
    print(f"Test loss : {test_loss:.4f}")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # TEXT GENERATION
    # ─────────────────────────────────────────────────────────────────────────

    prompts = [
        "The history of",
        "Scientists have discovered",
        "In the early twentieth century",
    ]

    print("=" * 60)
    print("Generated Text Samples")
    print("=" * 60)

    for prompt in prompts:
        output = model_eval.generate(prompt, max_new_tokens=120, temperature=0.7, top_k=50, top_p=0.9)
        print(f"\nPrompt : \"{prompt}\"")
        print(f"Output : {output.strip()}")
        print()

    # ─────────────────────────────────────────────────────────────────────────
    # INTERACTIVE MODE
    # ─────────────────────────────────────────────────────────────────────────

    print("=" * 60)
    print("Interactive Mode  (type 'quit' to exit)")
    print("=" * 60)

    while True:
        try:
            prompt = input("\nEnter a prompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt or prompt.lower() == "quit":
            break
        output = model_eval.generate(prompt, max_new_tokens=150, temperature=0.8, top_k=50, top_p=0.9)
        print(f"\n{output.strip()}")

    print("\nGoodbye!")

```
`prepare_data.py`:

```py
#!/usr/bin/env python3
"""
prepare_data.py
===============
Build tokenized binary files for training from FineWeb-Edu using streaming.

Outputs:
  data/train.bin
  data/val.bin
  data/test.bin

Dataset:
  HuggingFaceFW/fineweb-edu (streaming)
"""

import os
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm.auto import tqdm

# Local project cache for reproducibility and resume behavior.
os.environ.setdefault("HF_HOME", "./hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", "./hf_cache/datasets")
os.environ.setdefault("HF_HUB_CACHE", "./hf_cache/hub")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = "./hf_cache"
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = os.environ.get("DATASET_CONFIG", "sample-10BT")

# Stream only first N rows by default, matching your requested pattern.
MAX_EXAMPLES = int(os.environ.get("MAX_EXAMPLES", "1000000"))

# Deterministic split from one stream: 98% train, 1% val, 1% test.
TRAIN_FRAC = float(os.environ.get("TRAIN_FRAC", "0.98"))
VAL_FRAC = float(os.environ.get("VAL_FRAC", "0.01"))

# Flush chunks to disk to keep RAM bounded.
FLUSH_TOKENS = int(os.environ.get("FLUSH_TOKENS", "2000000"))

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token


def extract_text(row: dict) -> str:
    """Extract a usable text field across possible FineWeb-Edu schemas."""
    if "text" in row and isinstance(row["text"], str):
        return row["text"].strip()
    if "content" in row and isinstance(row["content"], str):
        return row["content"].strip()

    parts = []
    for key in ("prompt", "question", "instruction", "input", "answer", "response", "output"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())

    return "\n\n".join(parts).strip()


def encode_text(text: str):
    ids = enc.encode_ordinary(text)
    ids.append(EOT)
    return ids


def flush_tokens(fp, buffer_tokens):
    if not buffer_tokens:
        return 0
    arr = np.asarray(buffer_tokens, dtype=np.uint16)
    arr.tofile(fp)
    n = int(arr.size)
    buffer_tokens.clear()
    return n


def pick_split(i: int, total: int) -> str:
    train_cut = int(total * TRAIN_FRAC)
    val_cut = train_cut + int(total * VAL_FRAC)
    if i < train_cut:
        return "train"
    if i < val_cut:
        return "val"
    return "test"


if __name__ == "__main__":
    print("Loading FineWeb-Edu (streaming)...")

    # This follows your requested style while allowing MAX_EXAMPLES override.
    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="train",
        streaming=True,
        cache_dir=CACHE_DIR,
    )

    out_paths = {
        "train": DATA_DIR / "train.bin",
        "val": DATA_DIR / "val.bin",
        "test": DATA_DIR / "test.bin",
    }

    for p in out_paths.values():
        if p.exists():
            p.unlink()

    buffers = {"train": [], "val": [], "test": []}
    counts_examples = {"train": 0, "val": 0, "test": 0}
    counts_tokens = {"train": 0, "val": 0, "test": 0}

    with open(out_paths["train"], "ab") as f_train, open(out_paths["val"], "ab") as f_val, open(out_paths["test"], "ab") as f_test:
        fps = {"train": f_train, "val": f_val, "test": f_test}

        progress = tqdm(total=MAX_EXAMPLES, desc="Streaming+Encoding", unit="doc")
        for i, row in enumerate(dataset):
            if i >= MAX_EXAMPLES:
                break

            text = extract_text(row)
            if not text:
                progress.update(1)
                continue

            split = pick_split(i, MAX_EXAMPLES)
            toks = encode_text(text)
            buffers[split].extend(toks)
            counts_examples[split] += 1

            # Flush all splits together so val/test are written even if their
            # individual buffers never reach FLUSH_TOKENS (they're only 1% each).
            if len(buffers["train"]) >= FLUSH_TOKENS:
                for s in ("train", "val", "test"):
                    counts_tokens[s] += flush_tokens(fps[s], buffers[s])

            progress.update(1)

        progress.close()

        for split in ("train", "val", "test"):
            counts_tokens[split] += flush_tokens(fps[split], buffers[split])

    print("\nDone.")
    for split in ("train", "val", "test"):
        print(f"{split:>5}: {counts_examples[split]:>10,} docs  ->  {counts_tokens[split]:>12,} tokens")
    print(f"Saved files in: {DATA_DIR.resolve()}")

```
`push_to_hf.py`:

```py
#!/usr/bin/env python3
"""
Upload Tiny-GPT checkpoints to Hugging Face Hub.

Usage:
  python push_to_hf.py --repo-id yourname/Tiny-GPT
  python push_to_hf.py --repo-id yourname/Tiny-GPT --checkpoint checkpoints/best.pt

Auth:
  Set HF_TOKEN env var or run: huggingface-cli login
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Upload Tiny-GPT checkpoints to HF Hub")
    parser.add_argument("--repo-id", required=True, help="HF repo id, e.g. yourname/Tiny-GPT")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best.pt",
        help="Primary checkpoint path to upload (default: checkpoints/best.pt)",
    )
    parser.add_argument(
        "--latest-checkpoint",
        default="checkpoints/latest.pt",
        help="Optional latest checkpoint path to upload (default: checkpoints/latest.pt)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repo instead of public",
    )
    parser.add_argument(
        "--message",
        default="Upload Tiny-GPT checkpoints",
        help="Commit message for HF Hub",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token (or set HF_TOKEN env var)",
    )
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")

    try:
        from huggingface_hub import HfApi, upload_file
    except ImportError:
        print("[ERROR] Missing dependency: huggingface_hub")
        print("[ERROR] Install with: pip install huggingface_hub")
        sys.exit(1)

    checkpoint = Path(args.checkpoint)
    latest_checkpoint = Path(args.latest_checkpoint)

    if not checkpoint.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint}")
        sys.exit(1)

    api = HfApi(token=token)

    # Create repo if it does not exist yet.
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    print(f"Uploading {checkpoint} -> {args.repo_id}/best.pt")
    upload_file(
        path_or_fileobj=str(checkpoint),
        path_in_repo="best.pt",
        repo_id=args.repo_id,
        repo_type="model",
        token=token,
        commit_message=args.message,
    )

    if latest_checkpoint.exists():
        print(f"Uploading {latest_checkpoint} -> {args.repo_id}/latest.pt")
        upload_file(
            path_or_fileobj=str(latest_checkpoint),
            path_in_repo="latest.pt",
            repo_id=args.repo_id,
            repo_type="model",
            token=token,
            commit_message=args.message,
        )

    print("Done. Model checkpoints are now on Hugging Face Hub.")


if __name__ == "__main__":
    main()

```
`run.py`:

```py
"""
run.py – Inference script for MoE-GPT
========================================
Run the trained model anytime to generate text.

Usage:
    python run.py                  # Interactive mode
    python run.py --prompt "text"  # Generate from prompt
    python run.py --file data.txt  # Generate continuations from file

No training — just inference from the best checkpoint.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION (must match main.py)
# ═════════════════════════════════════════════════════════════════════════════

BLOCK_SIZE = 512
EMBED_DIM = 768
NUM_HEADS = 12
NUM_LAYERS = 12
NUM_EXPERTS = 8
TOP_K = 2
FFN_DIM = EMBED_DIM * 4
DROPOUT = 0.0
CHECKPOINT_DIR = "checkpoints"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# ═════════════════════════════════════════════════════════════════════════════
# 1. TOKENISER – GPT-2 BPE
# ═════════════════════════════════════════════════════════════════════════════

enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab  # 50,257


def encode(text: str) -> list:
    return enc.encode_ordinary(text)


def decode(ids: list) -> str:
    return enc.decode(ids)


def _infer_num_heads(embed_dim: int) -> int:
    """Infer a reasonable attention head count from embedding size."""
    for h in (16, 12, 8, 6, 4, 2, 1):
        if embed_dim % h == 0:
            return h
    return 1


def apply_model_config_from_state_dict(state_dict: dict):
    """Update global model hyperparameters to match checkpoint tensors."""
    global BLOCK_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, NUM_EXPERTS, FFN_DIM, vocab_size

    if "tok_emb.weight" not in state_dict or "pos_emb.weight" not in state_dict:
        return

    vocab_size = state_dict["tok_emb.weight"].shape[0]
    EMBED_DIM = state_dict["tok_emb.weight"].shape[1]
    BLOCK_SIZE = state_dict["pos_emb.weight"].shape[0]

    layer_ids = []
    for k in state_dict.keys():
        if k.startswith("blocks."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                layer_ids.append(int(parts[1]))
    if layer_ids:
        NUM_LAYERS = max(layer_ids) + 1

    router_key = "blocks.0.moe.router.weight"
    if router_key in state_dict:
        NUM_EXPERTS = state_dict[router_key].shape[0]

    ffn_key = "blocks.0.moe.experts.0.w1.weight"
    if ffn_key in state_dict:
        FFN_DIM = state_dict[ffn_key].shape[0]
    else:
        FFN_DIM = EMBED_DIM * 4

    NUM_HEADS = _infer_num_heads(EMBED_DIM)


def _get_model_state_from_checkpoint(ckpt: dict) -> dict:
    """Support both training checkpoint formats used in this repo."""
    if "model_state" in ckpt:
        return ckpt["model_state"]
    if "model" in ckpt:
        return ckpt["model"]
    raise KeyError("Checkpoint does not contain 'model_state' or 'model'")


def resolve_checkpoint_path(
    checkpoint_path=None,
    hf_repo=None,
    hf_filename="best.pt",
    hf_revision=None,
    hf_token=None,
):
    """Resolve a local checkpoint path, optionally downloading from HF Hub."""
    if hf_repo:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print("[ERROR] huggingface_hub is required for --hf-repo")
            print("[ERROR] Install it with: pip install huggingface_hub")
            sys.exit(1)

        cache_dir = Path("hf_cache") / "hub"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return hf_hub_download(
            repo_id=hf_repo,
            filename=hf_filename,
            revision=hf_revision,
            token=hf_token,
            cache_dir=str(cache_dir),
        )

    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best.pt")
    return checkpoint_path


# ═════════════════════════════════════════════════════════════════════════════
# 2. MODEL ARCHITECTURE (minimal — see main.py for full details)
# ═════════════════════════════════════════════════════════════════════════════


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = NUM_HEADS
        self.head_dim = EMBED_DIM // NUM_HEADS
        self.qkv = nn.Linear(EMBED_DIM, 3 * EMBED_DIM, bias=False)
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.attn_drop = nn.Dropout(DROPOUT)
        self.proj_drop = nn.Dropout(DROPOUT)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)).view(
                1, 1, BLOCK_SIZE, BLOCK_SIZE
            ),
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att.float(), dim=-1).to(x.dtype)
        att = self.attn_drop(att)

        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj_drop(self.proj(out))


class ExpertFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(EMBED_DIM, FFN_DIM)
        self.w2 = nn.Linear(FFN_DIM, EMBED_DIM)
        self.act = nn.GELU()
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        return self.drop(self.w2(self.act(self.w1(x))))


class MoELayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = nn.Linear(EMBED_DIM, NUM_EXPERTS, bias=False)
        self.experts = nn.ModuleList([ExpertFFN() for _ in range(NUM_EXPERTS)])

    def forward(self, x):
        B, T, C = x.shape
        flat = x.reshape(-1, C)
        N = flat.shape[0]

        logits = self.router(flat)
        probs = F.softmax(logits.float(), dim=-1)

        top_w, top_i = torch.topk(probs, TOP_K, dim=-1)
        top_w = (top_w / top_w.sum(dim=-1, keepdim=True)).to(x.dtype)

        out = torch.zeros_like(flat)
        for i, expert in enumerate(self.experts):
            mask = (top_i == i).any(dim=-1)
            if not mask.any():
                continue
            tokens = flat[mask]
            e_out = expert(tokens)
            match = (top_i[mask] == i).to(x.dtype)
            weights = (top_w[mask] * match).sum(-1, keepdim=True)
            out[mask] += weights * e_out

        return out.reshape(B, T, C)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(EMBED_DIM)
        self.moe = MoELayer()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.moe(self.ln2(x))
        return x


class MoEGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, EMBED_DIM)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(NUM_LAYERS)])
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(p)
        scale = (2 * NUM_LAYERS) ** -0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=0.02 * scale)
            for expert in block.moe.experts:
                nn.init.normal_(expert.w2.weight, mean=0.0, std=0.02 * scale)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(
            self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        )

        for block in self.blocks:
            x = block(x)

        logits = self.head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens=200,
        temperature=0.8,
        top_k=None,
        top_p=0.9,
    ):
        """
        Generate text from a prompt.

        Args:
            prompt: Starting text
            max_new_tokens: How many tokens to generate
            temperature: Higher = more random (0.5-1.5 typical)
            top_k: Keep only top-k most likely tokens (None = disabled)
            top_p: Nucleus sampling threshold (0.9 typical)
        """
        self.eval()
        ids = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)

        for _ in range(max_new_tokens):
            ctx = ids[:, -BLOCK_SIZE:]
            with torch.amp.autocast(
                "cuda", dtype=torch.bfloat16, enabled=(DTYPE == torch.bfloat16)
            ):
                logits, _ = self(ctx)
            logits = logits[:, -1, :].float() / temperature

            # Top-K filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Top-P (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumsum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumsum_probs > top_p
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            ids = torch.cat([ids, nxt], dim=1)

        self.train()
        return decode(ids[0].tolist())


# ═════════════════════════════════════════════════════════════════════════════
# 3. LOAD MODEL FROM CHECKPOINT
# ═════════════════════════════════════════════════════════════════════════════


def load_model(
    checkpoint_path=None,
    hf_repo=None,
    hf_filename="best.pt",
    hf_revision=None,
    hf_token=None,
):
    """Load the trained model from checkpoint."""
    checkpoint_path = resolve_checkpoint_path(
        checkpoint_path=checkpoint_path,
        hf_repo=hf_repo,
        hf_filename=hf_filename,
        hf_revision=hf_revision,
        hf_token=hf_token,
    )

    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found at: {checkpoint_path}")
        print(f"[ERROR] Have you run 'python main.py' yet?")
        sys.exit(1)

    print(f"Loading model from {checkpoint_path} ...", end=" ", flush=True)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model_state = _get_model_state_from_checkpoint(ckpt)
    apply_model_config_from_state_dict(model_state)

    model = MoEGPT()
    model = model.to(dtype=DTYPE, device=DEVICE)
    model.load_state_dict(model_state)
    model.eval()

    print("✓")
    print(f"  Device: {DEVICE.upper()}")
    print(f"  Dtype: {DTYPE}")
    print(
        f"  Model: block={BLOCK_SIZE}, emb={EMBED_DIM}, heads={NUM_HEADS}, "
        f"layers={NUM_LAYERS}, experts={NUM_EXPERTS}, ffn={FFN_DIM}"
    )
    print()

    return model


# ═════════════════════════════════════════════════════════════════════════════
# 4. INTERACTIVE & BATCH INFERENCE
# ═════════════════════════════════════════════════════════════════════════════


def interactive_mode(model):
    """Interactive text generation."""
    print("=" * 70)
    print("Interactive Mode – Type 'quit' to exit")
    print("=" * 70)
    print()
    print("Commands:")
    print("  quit          – Exit")
    print("  /temp 0.7     – Set temperature (default 0.8)")
    print("  /len 100      – Set max tokens (default 200)")
    print("  /topk 40      – Set top-k (default None = disabled)")
    print("  /topp 0.9     – Set top-p (default 0.9)")
    print()

    model.eval()
    temperature = 0.8
    max_tokens = 200
    top_k = None
    top_p = 0.9

    while True:
        try:
            user_input = input("Prompt > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            break

        # Handle commands
        if user_input.startswith("/"):
            parts = user_input.split()
            if len(parts) == 2:
                cmd, val = parts[0][1:], parts[1]
                try:
                    if cmd == "temp":
                        temperature = float(val)
                        print(f"Temperature set to {temperature}")
                    elif cmd == "len":
                        max_tokens = int(val)
                        print(f"Max tokens set to {max_tokens}")
                    elif cmd == "topk":
                        top_k = int(val)
                        print(f"Top-k set to {top_k}")
                    elif cmd == "topp":
                        top_p = float(val)
                        print(f"Top-p set to {top_p}")
                except ValueError:
                    print(f"Invalid value for {cmd}")
            continue

        print()
        with torch.no_grad():
            output = model.generate(
                user_input,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        print(output)
        print()

    model.train()
    print("\nGoodbye!")


def batch_generation(model, prompts, max_tokens=200, temperature=0.8):
    """Generate from a list of prompts."""
    print("=" * 70)
    print("Batch Generation")
    print("=" * 70)
    print()

    with torch.no_grad():
        for i, prompt in enumerate(prompts, 1):
            print(f"[{i}/{len(prompts)}] Prompt: {prompt}")
            output = model.generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )
            print(f"Output: {output}\n")


# ═════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ═════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using trained MoE-GPT model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                          # Interactive mode
  python run.py --prompt "Hello world"   # Generate from prompt
  python run.py --prompts file.txt       # Batch from file (one per line)
  python run.py --checkpoint custom.pt   # Use custom checkpoint
    python run.py --hf-repo user/Tiny-GPT  # Load from Hugging Face Hub
        """,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Single prompt to generate from",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        help="File with prompts (one per line) for batch generation",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (default: checkpoints/best.pt)",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help="Hugging Face repo id (e.g. user/Tiny-GPT). If set, download checkpoint from HF Hub.",
    )
    parser.add_argument(
        "--hf-filename",
        type=str,
        default="best.pt",
        help="Filename inside HF repo (default: best.pt)",
    )
    parser.add_argument(
        "--hf-revision",
        type=str,
        default=None,
        help="HF branch/tag/commit to download from",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HF token for private repos (or use HF_TOKEN env var)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Max tokens to generate (default: 200)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling (default: disabled)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p/nucleus sampling (default: 0.9)",
    )

    args = parser.parse_args()

    if args.hf_repo and args.checkpoint:
        print("[ERROR] Use either --checkpoint or --hf-repo, not both.")
        sys.exit(1)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # Load model
    model = load_model(
        checkpoint_path=args.checkpoint,
        hf_repo=args.hf_repo,
        hf_filename=args.hf_filename,
        hf_revision=args.hf_revision,
        hf_token=hf_token,
    )

    # Dispatch to appropriate mode
    if args.prompt:
        # Single prompt
        print(f"Prompt: {args.prompt}\n")
        model.eval()
        with torch.no_grad():
            output = model.generate(
                args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
        model.train()
        print(output)

    elif args.prompts:
        # Batch from file
        if not os.path.exists(args.prompts):
            print(f"[ERROR] File not found: {args.prompts}")
            sys.exit(1)
        with open(args.prompts) as f:
            prompts = [line.strip() for line in f if line.strip()]
        batch_generation(model, prompts, args.max_tokens, args.temperature)

    else:
        # Interactive mode
        interactive_mode(model)


if __name__ == "__main__":
    main()

```
`train_deepspeed.sh`:

```sh
#!/bin/bash
# train_deepspeed.sh - Launch training with DeepSpeed ZeRO-Infinity

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Tiny-GPT: DeepSpeed ZeRO-3 Training (CPU/NVMe Offloading)     ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Training mode: passive (default) or aggressive
TRAIN_MODE=$(echo "${TRAIN_MODE:-passive}" | tr '[:upper:]' '[:lower:]')
export TRAIN_MODE
if [ "$TRAIN_MODE" != "passive" ] && [ "$TRAIN_MODE" != "aggressive" ]; then
    echo -e "${YELLOW}!${NC} Invalid TRAIN_MODE='$TRAIN_MODE'. Use 'passive' or 'aggressive'."
    exit 1
fi

# Check DeepSpeed installation
echo -e "${BLUE}[1/4]${NC} Checking dependencies..."
python -c "import deepspeed" 2>/dev/null && echo -e "      ${GREEN}✓${NC} DeepSpeed installed" || {
    echo -e "      Installing DeepSpeed..."
    pip install deepspeed -q
    echo -e "      ${GREEN}✓${NC} DeepSpeed installed"
}
echo -e "      ${GREEN}✓${NC} All dependencies ready"
echo

# Checkpoint handling (keep by default for auto-resume)
echo -e "${BLUE}[2/4]${NC} Checkpoint handling..."
mkdir -p checkpoints
if [ "${RESET_CHECKPOINTS:-0}" = "1" ]; then
    rm -f checkpoints/*.pt
    echo -e "      ${GREEN}✓${NC} Checkpoints cleared (RESET_CHECKPOINTS=1)"
else
    if ls checkpoints/*.pt >/dev/null 2>&1; then
        echo -e "      ${GREEN}✓${NC} Existing checkpoints found (auto-resume enabled)"
    else
        echo -e "      ${GREEN}✓${NC} No existing checkpoints (fresh run)"
    fi
fi
echo

# Verify dataset
echo -e "${BLUE}[3/4]${NC} Verifying dataset..."
if [ -f "data/train.bin" ] && [ -f "data/val.bin" ] && [ -f "data/test.bin" ]; then
    echo -e "      ${GREEN}✓${NC} Dataset ready"
else
    echo -e "      ${YELLOW}!${NC} Dataset not found. Run: python prepare_data.py"
    exit 1
fi
echo

# Get number of GPUs
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -eq 0 ]; then
    NUM_GPUS=1
fi
export NUM_GPUS

# Build active DeepSpeed config based on TRAIN_MODE
python - <<'PY'
import json
import os
import multiprocessing

mode = os.environ.get("TRAIN_MODE", "passive").lower()
num_gpus = int(os.environ.get("NUM_GPUS", "1"))

with open("ds_config.json") as f:
    cfg = json.load(f)

zero = cfg.setdefault("zero_optimization", {})
off_opt = zero.setdefault("offload_optimizer", {"device": "cpu"})
act_ckpt = cfg.setdefault("activation_checkpointing", {})

if mode == "aggressive":
    # Higher-throughput profile: larger batches and no CPU checkpointing.
    cfg["train_micro_batch_size_per_gpu"] = 2
    cfg["gradient_accumulation_steps"] = 8
    cfg["train_batch_size"] = cfg["train_micro_batch_size_per_gpu"] * cfg["gradient_accumulation_steps"] * max(1, num_gpus)
    off_opt["pin_memory"] = True
    zero["reduce_bucket_size"] = 2e6
    act_ckpt["cpu_checkpointing"] = False
else:
    # Low-resource profile (current stable baseline).
    cfg["train_micro_batch_size_per_gpu"] = 1
    cfg["gradient_accumulation_steps"] = 4
    cfg["train_batch_size"] = cfg["train_micro_batch_size_per_gpu"] * cfg["gradient_accumulation_steps"] * max(1, num_gpus)
    off_opt["pin_memory"] = False
    zero["reduce_bucket_size"] = 1e6
    act_ckpt["cpu_checkpointing"] = True

with open("ds_config.active.json", "w") as f:
    json.dump(cfg, f, indent=2)
PY

# Show configuration
echo -e "${BLUE}[4/4]${NC} Launching DeepSpeed training..."
python -c "
import json
with open('ds_config.active.json') as f:
    cfg = json.load(f)
print('  DeepSpeed Configuration:')
print(f\"    • Mode: ${TRAIN_MODE}\")
print(f\"    • ZeRO Stage: {cfg['zero_optimization']['stage']}\")
print(f\"    • Optimizer Offload: {cfg['zero_optimization']['offload_optimizer']['device']}\")
param_offload = cfg['zero_optimization'].get('offload_param', {}).get('device', 'none')
print(f\"    • Parameter Offload: {param_offload}\")
print(f\"    • Mixed Precision: {'bfloat16' if cfg.get('bf16', {}).get('enabled') else 'float32'}\")
print(f\"    • Micro Batch: {cfg['train_micro_batch_size_per_gpu']}\")
print(f\"    • Grad Accum: {cfg['gradient_accumulation_steps']}\")
print(f\"    • Batch Size: {cfg['train_batch_size']}\")
print()
"

# Launch training with DeepSpeed
echo -e "${YELLOW}Starting DeepSpeed training...${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

# Skip CUDA version mismatch check (system CUDA >= PyTorch CUDA is fine)
export DS_SKIP_CUDA_CHECK=1
if [ "$TRAIN_MODE" = "aggressive" ]; then
    CPU_THREADS=$(nproc)
    export MAX_JOBS=$CPU_THREADS
    export OMP_NUM_THREADS=$CPU_THREADS
    export MKL_NUM_THREADS=$CPU_THREADS
    export ALLOW_TF32=1
    export USE_TORCH_COMPILE=${USE_TORCH_COMPILE:-0}
    export USE_ACTIVATION_CHECKPOINT=0
else
    export MAX_JOBS=1
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export ALLOW_TF32=1
    export USE_TORCH_COMPILE=${USE_TORCH_COMPILE:-0}
    export USE_ACTIVATION_CHECKPOINT=1
fi

# Main script reads this active config path.
export DS_CONFIG_PATH="ds_config.active.json"

# Launch with deepspeed
deepspeed --num_gpus $NUM_GPUS main_deepspeed.py

echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}Training complete!${NC}"
echo
echo "Check results:"
echo "  • Checkpoints: ls -lh checkpoints/"
echo "  • Generate: python run.py"
echo "  • Best model: checkpoints/best.pt"

```